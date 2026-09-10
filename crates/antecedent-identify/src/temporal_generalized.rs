//! `TemporalCpdag` / `TemporalPag` identification envelopes.
//!
//! CPDAG completions use temporal DAG adjustment. PAG completions retain mixed
//! endpoints and use visibility-aware MAG adjustment on a certified stationary
//! unfolding. Completions that do not identify contribute
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

/// Graph class and endpoint semantics of a temporal completion.
#[derive(Clone, Debug)]
pub enum TemporalCompletionGraph {
    /// A causally sufficient temporal DAG completion.
    Dag(TemporalDag),
    /// A directed/bidirected temporal MAG completion.
    Mag(TemporalPag),
}

/// Temporal identification cases with aligned coordinate maps.
#[derive(Clone, Debug)]
pub struct TemporalClassEnvelope {
    /// Per-completion identification and mass.
    pub envelope: IdentificationEnvelope<TemporalCompletionGraph>,
    /// Unfold indexers aligned with [`Self::envelope`].`cases`.
    pub indexers: Vec<TemporalIndexer>,
}

fn refuse_multi_step(query: &TemporalEffectQuery) -> Result<(), IdentificationError> {
    if !matches!(query.policy, TemporalPolicy::Pulse { .. })
        && !matches!(query.policy, TemporalPolicy::Sustained { from, until } if from == until)
    {
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
    /// Multi-step Sustained, unsupported marks, invalid stationary templates, or identification errors.
    pub fn identify_temporal_pag_envelope(
        &self,
        pag: &TemporalPag,
        query: &TemporalEffectQuery,
    ) -> Result<TemporalClassEnvelope, IdentificationError> {
        refuse_multi_step(query)?;
        let indexer = crate::temporal_mag::window(pag, query)?;
        let mut sampler = TemporalPagCompletionSampler::for_window(
            pag.clone(),
            self.config.max_completions,
            &indexer,
        )?;
        let mut cases = Vec::new();
        let mut unfolded = Vec::new();
        for completion in sampler.by_ref() {
            let (result, finite) = crate::temporal_mag::identify(
                &completion.graph,
                query,
                &indexer,
                self.config.max_candidates,
            )?;
            unfolded.push(finite);
            cases.push(GraphIdentificationCase {
                graph: TemporalCompletionGraph::Mag(completion.graph),
                result,
                weight: ProbabilityMass(self.config.per_completion_weight),
            });
        }
        let indexers = vec![indexer; cases.len()];
        let mut envelope = IdentificationEnvelope::from_cases(cases);
        let report = sampler.validation_report();
        let finite_audit = antecedent_graph::completion::audit_finite_mag_equivalence(&unfolded);
        envelope.push_features([GraphFeature { kind:Arc::from("temporal_pag_mixed_completion_audit"),detail:Arc::from(format!("stationary assignments={}, rejected_temporal_constraints={}, represented={}, ambiguous_class={}, window_audit_skipped={}, finite_query_equivalence={finite_audit:?}",report.assignments_examined,report.rejected_constraints,report.represented_completions,report.ambiguous_global_class,report.equivalence_audit_skipped)) }]);
        if finite_audit == Some(false) {
            return Err(IdentificationError::NotCertified {
                message: "temporal PAG endpoint refinements have different m-separation models in the unfolded query window",
            });
        }
        if sampler.hit_cap() {
            downgrade_capped(&mut envelope, self.config.max_completions);
        }
        if report.equivalence_audit_skipped || finite_audit.is_none() {
            envelope.push_features([GraphFeature { kind:Arc::from("temporal_pag_equivalence_audit_capped"),detail:Arc::from("global m-separation audit exceeded its budget; identification covers only retained, query-window-validated completions") }]);
            if envelope.status == IdentificationStatus::NonparametricallyIdentified {
                envelope.status = IdentificationStatus::PartiallyIdentified;
            }
        }
        Ok(TemporalClassEnvelope { envelope, indexers })
    }
}

fn identify_temporal_completions<I>(
    sampler: &mut I,
    query: &TemporalEffectQuery,
    id: &GeneralizedAdjustmentIdentifier,
) -> Result<
    (IdentificationEnvelope<TemporalCompletionGraph>, Vec<TemporalIndexer>),
    IdentificationError,
>
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
        cases.push(GraphIdentificationCase {
            graph: TemporalCompletionGraph::Dag(graph),
            result: identified.result,
            weight: w,
        });
        indexers.push(identified.indexer);
    }
    // Dense IDs include the history offset. Re-identify shorter unfoldings in
    // the shared window before comparing functionals or publishing an invariant.
    // Keep each arena intact: remapping just adjustment IDs would corrupt the
    // relationship between its expression and its temporal certificate.
    if let Some(history) = indexers.iter().map(TemporalIndexer::history).max() {
        for (case, indexer) in cases.iter_mut().zip(&mut indexers) {
            if indexer.history() != history {
                let TemporalCompletionGraph::Dag(graph) = &case.graph else {
                    unreachable!("this path only enumerates CPDAG completions");
                };
                let identified = backdoor.identify_temporal_with_history(graph, query, history)?;
                case.result = identified.result;
                *indexer = identified.indexer;
            }
        }
    }
    Ok((IdentificationEnvelope::from_cases(cases), indexers))
}

fn downgrade_capped(
    envelope: &mut IdentificationEnvelope<TemporalCompletionGraph>,
    max_completions: usize,
) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{DynamicRuleId, Lag, VariableId};

    #[test]
    fn completion_agreement_uses_a_shared_temporal_indexer() {
        struct Completion(TemporalDag);
        impl CompletionGraph for Completion {
            fn into_graph(self) -> TemporalDag {
                self.0
            }
        }
        let mut graph = TemporalDag::empty();
        let t = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let y = graph.add_lagged(VariableId::from_raw(1), Lag::from_raw(0)).unwrap();
        let z = graph.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
        graph.insert_directed(z, t).unwrap();
        graph.insert_directed(z, y).unwrap();
        graph.insert_directed(t, y).unwrap();
        let mut wider = graph.clone();
        wider.add_lagged(VariableId::from_raw(2), Lag::from_raw(2)).unwrap();
        let query =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_policy(TemporalPolicy::pulse(-1));
        let backdoor = TemporalBackdoorIdentifier::new();
        let first = backdoor.identify_temporal(&graph, &query).unwrap();
        let second = backdoor.identify_temporal(&wider, &query).unwrap();
        assert_ne!(first.indexer, second.indexer);
        let (envelope, indexers) = identify_temporal_completions(
            &mut vec![Completion(graph), Completion(wider)].into_iter(),
            &query,
            &GeneralizedAdjustmentIdentifier::new(),
        )
        .unwrap();
        assert_eq!(indexers[0], indexers[1]);
        assert!(envelope.invariant.is_some());
        let z = envelope.invariant.unwrap().adjustment_set[0];
        let key = indexers[0].key_of(z.raw()).unwrap();
        assert_eq!(key.variable, VariableId::from_raw(2));
        assert_eq!(key.offset, -1);
    }

    #[test]
    fn invisible_temporal_pag_retains_directed_and_latent_cases() {
        let mut pag = TemporalPag::empty();
        let t = pag.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let y = pag.add_lagged(VariableId::from_raw(1), Lag::from_raw(0)).unwrap();
        pag.insert_circle_arrow(t, y).unwrap();
        let mut query =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0);
        query.policy = TemporalPolicy::pulse(-1);
        let env = GeneralizedAdjustmentIdentifier::new()
            .identify_temporal_pag_envelope(&pag, &query)
            .unwrap()
            .envelope;
        assert_eq!(env.cases.len(), 2);
        assert_eq!(env.status, IdentificationStatus::NotIdentified);
        assert!(
            env.critical_graph_features
                .iter()
                .any(|f| f.kind.as_ref() == "temporal_pag_mixed_completion_audit")
        );
    }

    #[test]
    fn dynamic_schedule_is_not_silently_estimated_as_a_pulse() {
        let mut query =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0);
        query.policy = TemporalPolicy::dynamic(DynamicRuleId::from_raw(0), Arc::from([-1, 0]));
        assert!(refuse_multi_step(&query).is_err());
    }
}
