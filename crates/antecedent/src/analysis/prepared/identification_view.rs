//! Borrowed identification navigation. Inspection never compiles a contract or projects a
//! posterior onto a primary atom. Complete conversion is an explicit operation.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    CachedClassPosteriorAtomIdentification, CachedDbnPosteriorAtomIdentification,
    CachedTemporalClassPosteriorAtomIdentification, CachedTemporalHorizonIdentification,
    CachedTemporalIdentification, IdentificationCache, PreparedStudy,
};
use crate::planner::lowered::{IdentificationLayer, RetainedExecutionProof};
use antecedent_identify::{IdentificationResult, TemporalClassEnvelope};
use antecedent_prob::WeightedGraphSamples;
use serde_json::{Value, json};
use std::sync::Arc;

/// A bounded borrowed node in the complete retained identification hierarchy.
///
/// The lifetime is the prepared owner's. Native host adapters retain that owner
/// and resolve index paths without cloning proof arenas, case arrays or observations.
pub struct PreparedIdentificationView<'a> {
    study: &'a PreparedStudy,
    node: Node<'a>,
}

#[derive(Clone, Copy)]
enum Node<'a> {
    Root,
    Layer(&'a IdentificationLayer),
    Cache(&'a IdentificationCache),
    Proof(&'a IdentificationResult, Option<&'a antecedent_core::TemporalIndexer>, Option<f64>),
    Selected(&'a IdentificationResult, &'a antecedent_expr::IdentifiedEstimand),
    Temporal(&'a CachedTemporalIdentification),
    Horizon(&'a CachedTemporalHorizonIdentification),
    Pag(&'a antecedent_identify::IdentificationEnvelope<antecedent_graph::Pag>),
    Cpdag(&'a antecedent_identify::IdentificationEnvelope<antecedent_graph::Dag>),
    TemporalEnvelope(&'a TemporalClassEnvelope),
    PagCase(&'a antecedent_identify::GraphIdentificationCase<antecedent_graph::Pag>),
    DagCase(&'a antecedent_identify::GraphIdentificationCase<antecedent_graph::Dag>),
    TemporalCase(
        &'a antecedent_identify::GraphIdentificationCase<
            antecedent_identify::TemporalCompletionGraph,
        >,
        Option<&'a antecedent_core::TemporalIndexer>,
    ),
    Samples(&'a WeightedGraphSamples),
    Sample(&'a WeightedGraphSamples, usize),
    GraphAtoms(&'a [crate::analysis::identification_cache::CachedGraphPosteriorAtomIdentification]),
    GraphAtom(&'a crate::analysis::identification_cache::CachedGraphPosteriorAtomIdentification),
    ClassAtoms(&'a [CachedClassPosteriorAtomIdentification]),
    ClassAtom(&'a CachedClassPosteriorAtomIdentification),
    ClassCases(&'a CachedClassPosteriorAtomIdentification),
    ClassCase(&'a CachedClassPosteriorAtomIdentification, usize),
    DbnAtoms(&'a [CachedDbnPosteriorAtomIdentification]),
    DbnAtom(&'a CachedDbnPosteriorAtomIdentification),
    TemporalAtoms(&'a [CachedTemporalClassPosteriorAtomIdentification]),
    TemporalAtom(&'a CachedTemporalClassPosteriorAtomIdentification),
    Horizons(&'a [(u32, Arc<TemporalClassEnvelope>)]),
    HorizonEnvelope(u32, &'a TemporalClassEnvelope),
    Transport(&'a antecedent_identify::TransportIdentification),
    TransportCases(&'a [antecedent_identify::TransportIdentification]),
    Execution(&'a RetainedExecutionProof),
    Absent,
}

impl PreparedStudy {
    /// Read the complete retained identification hierarchy without compilation,
    /// fitting, row scanning, formula serialization or primary-atom projection.
    #[must_use]
    pub fn identification_view(&self) -> PreparedIdentificationView<'_> {
        PreparedIdentificationView { study: self, node: Node::Root }
    }
}

impl<'a> PreparedIdentificationView<'a> {
    /// Fixed-size metadata for this node. Composite products have no invented
    /// aggregate verdict, adjustment union or first-case assumption count.
    #[must_use]
    pub fn summary(&self) -> Value {
        let mut value = json!({"kind": self.node.kind(), "children": self.child_count(),
            "status": null, "status_unavailable": "composite_identification"});
        match self.node {
            Node::Root => {
                value["graph_class"] = json!(self.study.study().graph.class().as_str());
                value["cache_kind"] = json!(
                    self.study.study().identification.as_ref().map(IdentificationCache::kind)
                );
                value["max_completions"] = json!(self.study.study().max_completions);
                value["method"] = json!(
                    self.study
                        .plan()
                        .binding()
                        .identifier
                        .named()
                        .map(|identifier| identifier.as_str())
                );
                if let Some(plan) = self.study.plan().lowered() {
                    let layer = PreparedIdentificationView {
                        study: self.study,
                        node: Node::Layer(plan.identification_layer()),
                    }
                    .summary();
                    value["layer"] = layer["layer"].clone();
                    value["status_unavailable"] = layer["status_unavailable"].clone();
                    value["layer_details"] = layer;
                }
                if let Some(cache) = self.study.study().identification.as_ref() {
                    let retained =
                        PreparedIdentificationView { study: self.study, node: Node::Cache(cache) }
                            .summary();
                    value["status"] = retained["status"].clone();
                    value["status_unavailable"] = retained["status_unavailable"].clone();
                    value["status_domain"] = retained["status_domain"].clone();
                    value["retained"] = retained;
                }
            }
            Node::Layer(layer) => {
                value["layer"] = json!(match layer {
                    IdentificationLayer::Retained(_) => "retained",
                    IdentificationLayer::Deferred { .. } => "deferred",
                    IdentificationLayer::PerExecution { .. } => "per_execution",
                    IdentificationLayer::NonGraphical(_) => "non_graphical",
                    IdentificationLayer::NotClaimed { .. } => "not_claimed",
                });
                value["status_unavailable"] = json!(match layer {
                    IdentificationLayer::Retained(_) => "composite_identification",
                    IdentificationLayer::Deferred { .. } => "deferred_identification",
                    IdentificationLayer::PerExecution { .. } => "per_execution_identification",
                    IdentificationLayer::NonGraphical(_) => "non_graphical_identification",
                    IdentificationLayer::NotClaimed { .. } => "identification_not_claimed",
                });
                match layer {
                    IdentificationLayer::Deferred { reason, .. }
                    | IdentificationLayer::PerExecution { reason } => {
                        value["reason"] = json!(reason)
                    }
                    IdentificationLayer::NotClaimed { refusal } => {
                        value["refusal"] = json!(refusal)
                    }
                    IdentificationLayer::NonGraphical(proof) => {
                        value["non_graphical"] = json!(proof.as_str())
                    }
                    _ => {}
                }
            }
            Node::Cache(cache) => match cache {
                IdentificationCache::Static(c) => status(&mut value, c.identification.status),
                IdentificationCache::Pag(c) => envelope_summary(&mut value, &c.envelope),
                IdentificationCache::Cpdag(c) => envelope_summary(&mut value, &c.envelope),
                IdentificationCache::TemporalClass(c) => {
                    envelope_summary(&mut value, &c.envelope.envelope);
                    value["horizon_count"] = json!(c.by_horizon.len());
                }
                IdentificationCache::Temporal(c) => {
                    value["horizon_count"] = json!(c.by_horizon.len())
                }
                IdentificationCache::GraphPosterior(c) => {
                    value["sample_count"] = json!(c.graphs.n_samples);
                    value["atom_count"] = json!(c.atoms.len());
                    value["class_atom_count"] = json!(c.class_atoms.len());
                    value["mass_available"] = json!("per_sample");
                }
                IdentificationCache::DbnPosterior(c) => {
                    value["sample_count"] = json!(c.graphs.n_samples);
                    value["atom_count"] = json!(c.atoms.len());
                    value["horizon_demotion_count"] = json!(c.horizon_demotions.len());
                    value["mass_available"] = json!("per_sample");
                }
                IdentificationCache::TemporalClassPosterior(c) => {
                    value["sample_count"] = json!(c.graphs.n_samples);
                    value["class_atom_count"] = json!(c.class_atoms.len());
                    value["mass_available"] = json!("per_sample");
                }
                IdentificationCache::Transport(proof) => {
                    transport_outcome_summary(&mut value, proof)
                }
            },
            Node::Transport(proof) => transport_outcome_summary(&mut value, proof),
            Node::Proof(proof, indexer, weight) => {
                status(&mut value, proof.status);
                value["assumption_count"] = json!(proof.required_assumptions.len());
                value["derivation_step_count"] = json!(proof.derivation.steps.len());
                value["estimand_count"] = json!(proof.estimands.len());
                value["estimand_claim_count"] = json!(proof.estimand_claims.len());
                value["candidates_examined"] = json!(proof.performance.candidates_examined);
                value["sets_returned"] = json!(proof.performance.sets_returned);
                value["weight"] = json!(weight);
                value["temporal_namespace"] = indexer_value(indexer);
            }
            Node::Pag(envelope) => envelope_summary(&mut value, envelope),
            Node::Cpdag(envelope) => envelope_summary(&mut value, envelope),
            Node::TemporalEnvelope(envelope) | Node::HorizonEnvelope(_, envelope) => {
                envelope_summary(&mut value, &envelope.envelope)
            }
            Node::Horizon(entry) => {
                value["horizon"] = json!(entry.horizon);
                status(&mut value, entry.identification.status);
            }
            Node::ClassAtom(atom) => {
                value["key"] = json!(atom.key);
                status(&mut value, atom.identification.status);
                value["identified_weight"] = json!(atom.identified_weight);
                value["truncated_completions"] = json!(atom.truncated_completions);
            }
            Node::TemporalAtom(atom) => {
                value["key"] = json!(atom.key);
                status(&mut value, atom.identification.status);
                value["identified_weight"] = json!(atom.identified_weight);
                value["truncated_completions"] = json!(atom.truncated_completions);
            }
            Node::GraphAtom(atom) => {
                value["key"] = json!(atom.key);
                status(&mut value, atom.identification.status);
            }
            Node::DbnAtom(atom) => {
                value["key"] = json!(atom.key);
                status(&mut value, atom.identification.status);
            }
            Node::ClassCase(atom, index) => {
                let case = &atom.cases[index];
                status(&mut value, case.status);
                value["weight"] = json!(case.weight);
            }
            Node::PagCase(case) => {
                status(&mut value, case.result.status);
                value["weight"] = json!(case.weight.0);
            }
            Node::DagCase(case) => {
                status(&mut value, case.result.status);
                value["weight"] = json!(case.weight.0);
            }
            Node::TemporalCase(case, indexer) => {
                status(&mut value, case.result.status);
                value["weight"] = json!(case.weight.0);
                value["temporal_namespace"] = indexer_value(indexer);
            }
            Node::Absent => value["status_unavailable"] = json!("component_not_identified"),
            Node::Sample(graphs, index) => {
                value["key"] = json!(graphs.graph_keys.get(index));
                value["weight"] = json!(graphs.weights.get(index));
                value["identified"] = json!(
                    graphs
                        .identified
                        .get(index)
                        .map(|flag| matches!(flag, antecedent_prob::GraphIdentFlag::Identified))
                );
            }
            _ => {}
        }
        if let Node::HorizonEnvelope(horizon, _) = self.node {
            value["horizon"] = json!(horizon);
        }
        value
    }

    /// Number of immediate children, without enumerating or serializing them.
    #[must_use]
    pub fn child_count(&self) -> usize {
        match self.node {
            Node::Root => {
                usize::from(self.study.plan().lowered().is_some())
                    + usize::from(self.study.study().identification.is_some())
            }
            node => node.len(),
        }
    }

    /// Select one child by its retained positional coordinate. No siblings are
    /// constructed or serialized. The coordinate is local to this frozen owner.
    #[must_use]
    pub fn child(&self, index: usize) -> Option<Self> {
        let node = match self.node {
            Node::Root => {
                let layer = self
                    .study
                    .plan()
                    .lowered()
                    .map(|plan| Node::Layer(plan.identification_layer()));
                if index == 0 {
                    layer.or_else(|| self.study.study().identification.as_ref().map(Node::Cache))
                } else if index == 1 && layer.is_some() {
                    self.study.study().identification.as_ref().map(Node::Cache)
                } else {
                    None
                }
            }
            node => node.child(index),
        }?;
        Some(Self { study: self.study, node })
    }

    fn selected_proof(
        &self,
    ) -> Result<
        (
            &'a IdentificationResult,
            Option<&'a antecedent_core::TemporalIndexer>,
            Option<&'a antecedent_expr::IdentifiedEstimand>,
        ),
        crate::CausalError,
    > {
        match self.node {
            Node::Proof(proof, indexer, _) => Ok((proof, indexer, None)),
            Node::Selected(proof, selected) => Ok((proof, None, Some(selected))),
            _ => Err(crate::unsupported_reason!(
                "option_not_applicable",
                "identification.proof_selection_required: select one retained proof node; composite products have no implicit first proof"
            )),
        }
    }

    fn select_estimand(
        &self,
        requested: Option<usize>,
    ) -> Result<
        Option<(
            &'a antecedent_expr::IdentifiedEstimand,
            Option<&'a antecedent_core::TemporalIndexer>,
            usize,
        )>,
        crate::CausalError,
    > {
        let (proof, indexer, selected) = self.selected_proof()?;
        let index = match requested {
            Some(index) => index,
            None if selected.is_some() => {
                let selected = selected.expect("selected estimand checked above");
                proof.estimands.iter().position(|candidate| candidate == selected).ok_or_else(
                    || crate::CausalError::Compile {
                        message: "selected estimand is absent from its retained proof".into(),
                    },
                )?
            }
            None if proof.estimands.len() == 1 => 0,
            None if proof.estimands.is_empty() => return Ok(None),
            None => {
                return Err(crate::unsupported_reason!(
                    "option_not_applicable",
                    "identification.estimand_selection_required: this proof lists several estimands; supply an explicit estimand index"
                ));
            }
        };
        let estimand = proof.estimands.get(index).ok_or_else(|| crate::CausalError::Compile {
            message: "identification estimand index is out of range".into(),
        })?;
        Ok(Some((estimand, indexer, index)))
    }

    /// Bounded descriptions of the alternatives on one explicitly selected proof.
    /// Functional IDs remain scoped to this proof's native arena; no expression is converted.
    ///
    /// # Errors
    /// A composite node was not explicitly selected or the page exceeds 256 entries.
    pub fn estimands_page(&self, offset: usize, limit: usize) -> Result<Value, crate::CausalError> {
        let (proof, _, selected) = self.selected_proof()?;
        let end = accessor_page_end(proof.estimands.len(), offset, limit)?;
        let items = proof.estimands.get(offset..end).unwrap_or(&[]).iter().enumerate().map(|(local,e)| {
            let index=offset+local;
            json!({"index":index,"selected":selected.map(|chosen|chosen==e),"method":e.method,"functional":e.functional.raw(),
                "adjustment_count":e.adjustment_set.len(),"instrument_count":e.instruments.len(),
                "mediator_count":e.mediators.len(),"has_rd_design":e.rd_design.is_some(),
                "claim_status":if proof.estimand_claims.is_empty() {Some(proof.status.as_str())} else {proof.estimand_claims.get(index).map(|c|c.status.as_str())},
                "claim_basis":if proof.estimand_claims.is_empty() {"shared_native_proof"} else {"per_estimand_claim"},
                "claim_unavailable":(!proof.estimand_claims.is_empty() && proof.estimand_claims.get(index).is_none()).then_some("estimand_claim_not_retained")})
        }).collect::<Vec<_>>();
        Ok(accessor_page(proof.estimands.len(), offset, end, items))
    }

    /// Named adjustment coordinates for an explicitly chosen estimand. A temporal
    /// proof requires its retained indexer; native dense IDs are never schema guesses.
    ///
    /// # Errors
    /// A composite/ambiguous selection, invalid coordinate, or oversized page.
    pub fn adjustment_page(
        &self,
        estimand: Option<usize>,
        offset: usize,
        limit: usize,
    ) -> Result<Value, crate::CausalError> {
        accessor_page_end(0, offset, limit)?;
        let Some((estimand, indexer, index)) = self.select_estimand(estimand)? else {
            return Ok(json!({"available":false,"reason":"no_retained_estimand"}));
        };
        let temporal = self.study.study().graph.class().as_str().starts_with("temporal")
            || estimand.method.starts_with("temporal.")
            || estimand.method.starts_with("temporal_mediation.")
            || matches!(self.study.query(), antecedent_core::CausalQuery::TemporalEffect(_))
            || matches!(self.study.query(),antecedent_core::CausalQuery::Response(q) if q.temporal.is_some());
        if indexer.is_none() && temporal {
            return Ok(
                json!({"available":false,"reason":"temporal_namespace_not_retained","estimand":index}),
            );
        }
        let end = accessor_page_end(estimand.adjustment_set.len(), offset, limit)?;
        let items=estimand.adjustment_set.get(offset..end).unwrap_or(&[]).iter().map(|id| {
            let key=match indexer {
                Some(indexer)=>indexer.key_of(id.raw()).map_err(|error|crate::CausalError::Compile{message:error.to_string()})?,
                None=>antecedent_core::TemporalNodeKey::contemporaneous(*id),
            };
            let name=self.study.schema().variables().get(key.variable.as_usize()).ok_or_else(||crate::CausalError::Compile {
                message:"identification coordinate is outside its retained schema".into(),
            })?.name.as_ref();
            Ok(json!({"name":name,"variable":key.variable.raw(),"offset":key.offset,"node":id.raw()}))
        }).collect::<Result<Vec<_>,crate::CausalError>>()?;
        let mut page = accessor_page(estimand.adjustment_set.len(), offset, end, items);
        page["estimand"] = json!(index);
        page["temporal_namespace"] = indexer_value(indexer);
        Ok(page)
    }

    /// Ordered native assumptions for this proof, or the exact claim of a selected
    /// estimand. No claim falls back to the union of other estimands' assumptions.
    ///
    /// # Errors
    /// An invalid selection or a page exceeding 256 records.
    pub fn assumptions_page(
        &self,
        estimand: Option<usize>,
        offset: usize,
        limit: usize,
    ) -> Result<Value, crate::CausalError> {
        let (proof, _, selected) = self.selected_proof()?;
        accessor_page_end(0, offset, limit)?;
        let estimand = if estimand.is_none() && selected.is_some() {
            self.select_estimand(None)?.map(|(_, _, index)| index)
        } else {
            estimand
        };
        let (assumptions, scope, status) = if let Some(index) = estimand {
            if proof.estimands.get(index).is_none() {
                return Err(crate::CausalError::Compile {
                    message: "identification estimand index is out of range".into(),
                });
            }
            if proof.estimand_claims.is_empty() {
                // IdentificationResult explicitly defines an empty claim vector
                // as all estimands sharing the proof's status and assumptions.
                (&proof.required_assumptions, "estimand_shared_native_proof", proof.status)
            } else {
                let Some(claim) = proof.estimand_claims.get(index) else {
                    return Ok(
                        json!({"available":false,"reason":"estimand_claim_not_retained","estimand":index}),
                    );
                };
                (&claim.required_assumptions, "estimand", claim.status)
            }
        } else {
            (&proof.required_assumptions, "proof", proof.status)
        };
        let end = accessor_page_end(assumptions.entries.len(), offset, limit)?;
        // The existing lossless encoder is applied only to the requested page.
        let selected = antecedent_core::AssumptionSet {
            entries: assumptions.entries.get(offset..end).unwrap_or(&[]).to_vec(),
        };
        let encoded = antecedent_io::assumptions_to_wire(&selected);
        let items = encoded
            .iter()
            .map(|entry| serde_json::to_value(entry).map_err(serialization))
            .collect::<Result<Vec<_>, _>>()?;
        let mut page = accessor_page(assumptions.entries.len(), offset, end, items);
        page["scope"] = json!(scope);
        page["status"] = json!(status.as_str());
        page["estimand"] = json!(estimand);
        Ok(page)
    }

    /// Ordered native derivation steps on this explicitly selected proof.
    ///
    /// # Errors
    /// A composite selection or a page exceeding 256 records.
    pub fn derivation_page(
        &self,
        offset: usize,
        limit: usize,
    ) -> Result<Value, crate::CausalError> {
        let (proof, _, _) = self.selected_proof()?;
        let end = accessor_page_end(proof.derivation.steps.len(), offset, limit)?;
        let items = proof
            .derivation
            .steps
            .get(offset..end)
            .unwrap_or(&[])
            .iter()
            .map(|step| json!({"rule":step.rule,"detail":step.detail}))
            .collect();
        Ok(accessor_page(proof.derivation.steps.len(), offset, end, items))
    }

    /// Complete conversion of this subtree with no caller context, for unit tests of the
    /// converter; hosts use [`Self::materialize_with_context`].
    ///
    /// # Errors
    /// A retained proof cannot be encoded by the native wire converter.
    #[cfg(test)]
    pub(crate) fn materialize(&self) -> Result<Value, crate::CausalError> {
        self.materialize_inner(None)
    }

    /// Explicit conversion under a caller's cancellation authority. Scientific
    /// computation is never performed. Allocation admission for the complete
    /// JSON tree remains the materializing caller's responsibility.
    ///
    /// # Errors
    /// Caller cancellation or a native wire conversion failure.
    pub fn materialize_with_context(
        &self,
        context: &antecedent_core::ExecutionContext,
    ) -> Result<Value, crate::CausalError> {
        self.materialize_inner(Some(context))
    }

    fn materialize_inner(
        &self,
        context: Option<&antecedent_core::ExecutionContext>,
    ) -> Result<Value, crate::CausalError> {
        let cancelled = || {
            if context.is_some_and(|context| context.cancellation.is_cancelled()) {
                Err(crate::CausalError::Cancelled { stage: "identification_materialize" })
            } else {
                Ok(())
            }
        };
        cancelled()?;
        let mut value = json!({"summary": self.summary()});
        match self.node {
            Node::Proof(proof, _, _) => {
                value["identification"] =
                    serde_json::to_value(antecedent_io::identification_to_wire_with_registry(
                        proof,
                        self.study.population_registry(),
                    )?)
                    .map_err(serialization)?
            }
            Node::Transport(proof) => {
                value["transport"] =
                    serde_json::to_value(antecedent_io::transport_identification_to_wire(proof))
                        .map_err(serialization)?
            }
            Node::Samples(samples) => {
                value["edge_marginals"] = json!(samples.edge_marginals);
                value["orientation_marginals"] = json!(samples.orientation_marginals);
            }
            Node::Layer(IdentificationLayer::NonGraphical(proof)) => {
                value["non_graphical"] = non_graphical(proof)
            }
            Node::PagCase(case) => {
                value["graph"] = serde_json::to_value(antecedent_io::pag_to_wire(&case.graph)?)
                    .map_err(serialization)?
            }
            Node::DagCase(case) => {
                value["graph"] = serde_json::to_value(antecedent_io::dag_to_wire(&case.graph)?)
                    .map_err(serialization)?
            }
            Node::TemporalCase(case, _) => {
                value["fingerprint"] = json!(case.graph.fingerprint());
                value["graph"] = temporal_graph(&case.graph)?;
            }
            Node::Horizon(entry) => value["selected_estimand"] = estimand(&entry.estimand),
            Node::Selected(_, selected) => value["selected_estimand"] = estimand(selected),
            Node::ClassCase(atom, index) => {
                value["selected_estimand"] =
                    atom.cases[index].estimand.as_ref().map_or(Value::Null, estimand)
            }
            Node::ClassAtom(atom) => {
                value["invariant"] = atom.invariant.as_ref().map_or(Value::Null, estimand)
            }
            Node::TemporalAtom(atom) => {
                value["invariant"] = atom.invariant.as_ref().map_or(Value::Null, estimand)
            }
            Node::GraphAtom(atom) => value["selected_estimand"] = estimand(&atom.estimand),
            Node::DbnAtom(atom) => value["selected_estimand"] = estimand(&atom.estimand),
            Node::Cache(IdentificationCache::Static(cache)) => {
                value["selected_estimand"] = estimand(&cache.estimand)
            }
            Node::Pag(envelope) => envelope_details(&mut value, envelope),
            Node::Cpdag(envelope) => envelope_details(&mut value, envelope),
            Node::TemporalEnvelope(envelope) | Node::HorizonEnvelope(_, envelope) => {
                envelope_details(&mut value, &envelope.envelope)
            }
            Node::Cache(IdentificationCache::DbnPosterior(cache)) => {
                value["demotion"] = demotion(&cache.identify_demotion);
                value["horizon_demotions"] = Value::Array(
                    cache
                        .horizon_demotions
                        .iter()
                        .map(|(h, d)| json!({"horizon":h,"demotion":demotion(d)}))
                        .collect(),
                );
            }
            _ => {}
        }
        let mut children = Vec::new();
        for index in 0..self.child_count() {
            let child = self.child(index).ok_or_else(|| crate::CausalError::Compile {
                message: "retained identification child is missing".into(),
            })?;
            children.push(child.materialize_inner(context)?);
        }
        value["children"] = Value::Array(children);
        cancelled()?;
        Ok(value)
    }
}

fn accessor_page_end(
    length: usize,
    offset: usize,
    limit: usize,
) -> Result<usize, crate::CausalError> {
    if limit > 256 {
        return Err(crate::CausalError::Compile {
            message: "identification page limit must be at most 256".into(),
        });
    }
    Ok(offset.saturating_add(limit).min(length))
}
fn accessor_page(total: usize, offset: usize, end: usize, items: Vec<Value>) -> Value {
    json!({"available":true,"total":total,"offset":offset,"items":items,"next_offset":(end<total).then_some(end)})
}
fn transport_outcome_summary(
    value: &mut Value,
    proof: &antecedent_identify::TransportIdentification,
) {
    let outcome = proof.outcome();
    value["status"] = json!(outcome.kind.as_str());
    value["status_domain"] = json!("transport_outcome");
    value["status_unavailable"] = Value::Null;
    value["reason"] = json!(outcome.reason);
    value["location_factor"] = json!(outcome.location.as_ref().and_then(|l| l.factor.as_ref()));
    value["location_node_count"] =
        json!(outcome.location.as_ref().map_or(0, |l| l.graph_nodes.len()));
}

fn serialization(error: serde_json::Error) -> crate::CausalError {
    crate::CausalError::Compile { message: error.to_string() }
}
fn status(value: &mut Value, status: antecedent_core::IdentificationStatus) {
    value["status"] = json!(status.as_str());
    value["status_domain"] = json!("identification");
    value["status_unavailable"] = Value::Null;
}
fn envelope_summary<G>(
    value: &mut Value,
    envelope: &antecedent_identify::IdentificationEnvelope<G>,
) {
    status(value, envelope.status);
    value["identified_weight"] = json!(envelope.identified_weight.0);
    value["unidentified_weight"] = json!(envelope.unidentified_weight.0);
    value["truncated_completions"] = json!(envelope.truncated_completions);
    value["case_count"] = json!(envelope.cases.len());
    value["has_invariant"] = json!(envelope.invariant.is_some());
}
fn envelope_details<G>(
    value: &mut Value,
    envelope: &antecedent_identify::IdentificationEnvelope<G>,
) {
    value["invariant"] = envelope.invariant.as_ref().map_or(Value::Null, estimand);
    value["critical_graph_features"] = json!(
        envelope
            .critical_graph_features
            .iter()
            .map(|feature| json!({"kind":feature.kind,"detail":feature.detail}))
            .collect::<Vec<_>>()
    );
}
fn estimand(estimand: &antecedent_expr::IdentifiedEstimand) -> Value {
    let variables = |variables: &[antecedent_core::VariableId]| {
        variables.iter().map(|id| id.raw()).collect::<Vec<_>>()
    };
    json!({"method":estimand.method,"adjustment_set":variables(&estimand.adjustment_set),
        "instruments":variables(&estimand.instruments),"mediators":variables(&estimand.mediators),
        "functional":estimand.functional.raw(),"rd_design":estimand.rd_design.map(|rd|json!({"running_variable":rd.running_variable.raw(),"cutoff":rd.cutoff,"bandwidth":rd.bandwidth}))})
}
fn temporal_graph(
    graph: &antecedent_identify::TemporalCompletionGraph,
) -> Result<Value, crate::CausalError> {
    match graph {
        antecedent_identify::TemporalCompletionGraph::Dag(graph) => {
            Ok(json!({"kind":"temporal_dag","graph":antecedent_io::temporal_dag_to_wire(graph)?}))
        }
        antecedent_identify::TemporalCompletionGraph::Mag(graph) => {
            let nodes = graph.nodes().iter().map(|node| match node {
                antecedent_core::NodeRef::Static(variable) => json!({"kind":"static","variable":variable.raw()}),
                antecedent_core::NodeRef::Lagged {variable,lag} => json!({"kind":"lagged","variable":variable.raw(),"lag":lag.raw()}),
                antecedent_core::NodeRef::Unfolded {variable,offset} => json!({"kind":"unfolded","variable":variable.raw(),"offset":offset}),
                antecedent_core::NodeRef::Context {variable,environment} => json!({"kind":"context","variable":variable.raw(),"environment":environment.map(|e|e.raw())}),
            }).collect::<Vec<_>>();
            let endpoint = |mark| match mark {
                antecedent_graph::Endpoint::Tail => "tail",
                antecedent_graph::Endpoint::Arrow => "arrow",
                antecedent_graph::Endpoint::Circle => "circle",
                antecedent_graph::Endpoint::Conflict => "conflict",
            };
            let edges = graph.edges().iter().map(|edge| json!({"a":edge.a.raw(),"b":edge.b.raw(),
                "at_a":endpoint(edge.at_a),"at_b":endpoint(edge.at_b),"middle":match edge.middle {
                    antecedent_graph::MiddleMark::Unknown=>"unknown",antecedent_graph::MiddleMark::Left=>"left",
                    antecedent_graph::MiddleMark::Right=>"right",antecedent_graph::MiddleMark::Both=>"both",antecedent_graph::MiddleMark::Empty=>"empty",
                }})).collect::<Vec<_>>();
            Ok(json!({"kind":"temporal_mag","nodes":nodes,"edges":edges}))
        }
    }
}
fn indexer_value(indexer: Option<&antecedent_core::TemporalIndexer>) -> Value {
    indexer.map_or(Value::Null, |i| json!({"variable_count":i.variable_count(),"history":i.history(),"horizon":i.horizon()}))
}
fn demotion(d: &crate::analysis::identification_cache::DbnIdentifyDemotion) -> Value {
    json!({"invalid_graph":d.invalid_graph,"identify_failed":d.identify_failed,"not_identified":d.not_identified,"no_estimand":d.no_estimand})
}
fn non_graphical(proof: &crate::planner::lowered::NonGraphicalIdentification) -> Value {
    use crate::planner::lowered::NonGraphicalIdentification::*;
    match proof {
        SpecifiedModelConditional { model } => {
            json!({"kind":"specified_model_conditional","model":model})
        }
        ExperimentalRecovery { design, unresolved } => {
            json!({"kind":"experimental_recovery","design":design,"unresolved":unresolved.iter().map(|(target,missing)|json!({"target":target.raw(),"missing":missing.iter().map(|v|v.raw()).collect::<Vec<_>>()})).collect::<Vec<_>>()})
        }
        CyclicIoScm { source } => json!({"kind":"cyclic_io_scm","source":source}),
        SharedEvidenceContract { contract } => {
            json!({"kind":"shared_evidence_contract","contract":contract})
        }
    }
}

impl<'a> Node<'a> {
    fn kind(self) -> &'static str {
        match self {
            Self::Root => "identification",
            Self::Layer(_) => "layer",
            Self::Cache(c) => c.kind(),
            Self::Proof(..) => "proof",
            Self::Selected(..) => "selected_proof",
            Self::Temporal(_) => "horizons",
            Self::Horizon(_) | Self::HorizonEnvelope(..) => "horizon",
            Self::PagCase(_) | Self::DagCase(_) | Self::TemporalCase(..) => "completion_case",
            Self::Pag(_) | Self::Cpdag(_) | Self::TemporalEnvelope(_) => "completion_envelope",
            Self::Samples(_) => "posterior_samples",
            Self::Sample(..) => "posterior_sample",
            Self::GraphAtoms(_) | Self::DbnAtoms(_) => "identified_atoms",
            Self::ClassAtoms(_) | Self::TemporalAtoms(_) => "class_atoms",
            Self::GraphAtom(_) | Self::DbnAtom(_) => "identified_atom",
            Self::ClassAtom(_) | Self::TemporalAtom(_) => "class_atom",
            Self::ClassCases(_) => "completion_cases",
            Self::ClassCase(..) => "completion_case",
            Self::Horizons(_) => "horizon_envelopes",
            Self::Transport(_) => "transport",
            Self::TransportCases(_) => "transport_cases",
            Self::Execution(_) => "execution_proofs",
            Self::Absent => "absent_component",
        }
    }
    fn len(self) -> usize {
        match self {
            Self::Root => 0, // The owner-dependent root count is supplied below.
            Self::Layer(IdentificationLayer::Retained(retained)) => {
                retained.caches().len() + usize::from(retained.execution_proof().is_some())
            }
            Self::Layer(IdentificationLayer::Deferred { failure, .. }) => {
                usize::from(failure.is_some())
            }
            Self::Cache(c) => match c {
                IdentificationCache::Static(_) => 1,
                IdentificationCache::Pag(c) => 2 + usize::from(c.transport.is_some()),
                IdentificationCache::Cpdag(c) => 2 + usize::from(c.transport.is_some()),
                IdentificationCache::Temporal(_) | IdentificationCache::Transport(_) => 1,
                IdentificationCache::TemporalClass(_) => 2,
                IdentificationCache::GraphPosterior(_) => 3,
                IdentificationCache::DbnPosterior(_)
                | IdentificationCache::TemporalClassPosterior(_) => 2,
            },
            Self::Temporal(c) => c.by_horizon.len(),
            Self::Horizon(_) => 1,
            Self::Pag(e) => e.cases.len(),
            Self::Cpdag(e) => e.cases.len(),
            Self::TemporalEnvelope(e) | Self::HorizonEnvelope(_, e) => e.envelope.cases.len(),
            Self::Samples(g) => g.n_samples,
            Self::GraphAtoms(a) => a.len(),
            Self::ClassAtoms(a) => a.len(),
            Self::ClassCases(a) => a.cases.len(),
            Self::DbnAtoms(a) => a.len(),
            Self::TemporalAtoms(a) => a.len(),
            Self::Horizons(h) => h.len(),
            Self::TransportCases(c) => c.len(),
            Self::PagCase(_) | Self::DagCase(_) | Self::TemporalCase(..) => 1,
            Self::Selected(..) | Self::ClassCase(..) => 1,
            Self::GraphAtom(a) => 1 + usize::from(a.transport.is_some()),
            Self::ClassAtom(_) => 2,
            Self::DbnAtom(a) => 1 + usize::from(a.horizons.is_some()),
            Self::TemporalAtom(_) => 2,
            Self::Execution(p) => match p {
                RetainedExecutionProof::Admg(c) => c.len(),
                RetainedExecutionProof::Pag { envelopes } => envelopes.len(),
                _ => 1,
            },
            _ => 0,
        }
    }
    fn child(self, index: usize) -> Option<Self> {
        match self {
            Self::Layer(IdentificationLayer::Retained(r)) => {
                r.caches().get(index).map(Self::Cache).or_else(|| {
                    (index == r.caches().len())
                        .then(|| r.execution_proof().map(Self::Execution))
                        .flatten()
                })
            }
            Self::Layer(IdentificationLayer::Deferred { failure, .. }) if index == 0 => {
                failure.as_deref().map(|r| Self::Proof(r, None, None))
            }
            Self::Cache(c) => match (c, index) {
                (IdentificationCache::Static(c), 0) => {
                    Some(Self::Proof(&c.identification, None, None))
                }
                (IdentificationCache::Pag(c), 0) => {
                    Some(Self::Proof(&c.identification, None, None))
                }
                (IdentificationCache::Pag(c), 1) => Some(Self::Pag(&c.envelope)),
                (IdentificationCache::Pag(c), 2) => {
                    c.transport.as_deref().map(Self::TransportCases)
                }
                (IdentificationCache::Cpdag(c), 0) => {
                    Some(Self::Proof(&c.identification, None, None))
                }
                (IdentificationCache::Cpdag(c), 1) => Some(Self::Cpdag(&c.envelope)),
                (IdentificationCache::Cpdag(c), 2) => {
                    c.transport.as_deref().map(Self::TransportCases)
                }
                (IdentificationCache::Temporal(c), 0) => Some(Self::Temporal(c)),
                (IdentificationCache::TemporalClass(c), 0) => {
                    Some(Self::TemporalEnvelope(&c.envelope))
                }
                (IdentificationCache::TemporalClass(c), 1) => Some(Self::Horizons(&c.by_horizon)),
                (IdentificationCache::Transport(c), 0) => Some(Self::Transport(c)),
                (IdentificationCache::GraphPosterior(c), 0) => Some(Self::Samples(&c.graphs)),
                (IdentificationCache::GraphPosterior(c), 1) => Some(Self::GraphAtoms(&c.atoms)),
                (IdentificationCache::GraphPosterior(c), 2) => {
                    Some(Self::ClassAtoms(&c.class_atoms))
                }
                (IdentificationCache::DbnPosterior(c), 0) => Some(Self::Samples(&c.graphs)),
                (IdentificationCache::DbnPosterior(c), 1) => Some(Self::DbnAtoms(&c.atoms)),
                (IdentificationCache::TemporalClassPosterior(c), 0) => {
                    Some(Self::Samples(&c.graphs))
                }
                (IdentificationCache::TemporalClassPosterior(c), 1) => {
                    Some(Self::TemporalAtoms(&c.class_atoms))
                }
                _ => None,
            },
            Self::Temporal(c) => c.by_horizon.get(index).map(Self::Horizon),
            Self::Horizon(c) if index == 0 => {
                Some(Self::Proof(&c.identification, Some(&c.indexer), None))
            }
            Self::Selected(proof, _) if index == 0 => Some(Self::Proof(proof, None, None)),
            Self::Pag(e) => e.cases.get(index).map(Self::PagCase),
            Self::Cpdag(e) => e.cases.get(index).map(Self::DagCase),
            Self::TemporalEnvelope(e) | Self::HorizonEnvelope(_, e) => {
                e.envelope.cases.get(index).map(|c| Self::TemporalCase(c, e.indexers.get(index)))
            }
            Self::PagCase(c) if index == 0 => Some(Self::Proof(&c.result, None, Some(c.weight.0))),
            Self::DagCase(c) if index == 0 => Some(Self::Proof(&c.result, None, Some(c.weight.0))),
            Self::TemporalCase(c, i) if index == 0 => {
                Some(Self::Proof(&c.result, i, Some(c.weight.0)))
            }
            Self::Samples(g) if index < g.n_samples => Some(Self::Sample(g, index)),
            Self::GraphAtoms(a) => a.get(index).map(Self::GraphAtom),
            Self::ClassAtoms(a) => a.get(index).map(Self::ClassAtom),
            Self::ClassCases(a) => (index < a.cases.len()).then_some(Self::ClassCase(a, index)),
            Self::ClassCase(atom, case) if index == 0 => Some(Self::Proof(
                &atom.cases[case].identification,
                None,
                Some(atom.cases[case].weight),
            )),
            Self::DbnAtoms(a) => a.get(index).map(Self::DbnAtom),
            Self::TemporalAtoms(a) => a.get(index).map(Self::TemporalAtom),
            Self::Horizons(h) => h.get(index).map(|(h, e)| Self::HorizonEnvelope(*h, e)),
            Self::TransportCases(c) => c.get(index).map(Self::Transport),
            Self::GraphAtom(a) if index == 0 => Some(Self::Proof(&a.identification, None, None)),
            Self::GraphAtom(a) if index == 1 => a.transport.as_deref().map(Self::Transport),
            Self::ClassAtom(a) if index == 0 => Some(Self::Proof(&a.identification, None, None)),
            Self::ClassAtom(a) if index == 1 => Some(Self::ClassCases(a)),
            Self::DbnAtom(a) if index == 0 => {
                Some(Self::Proof(&a.identification, Some(&a.indexer), None))
            }
            Self::DbnAtom(a) if index == 1 => a.horizons.as_ref().map(Self::Temporal),
            Self::TemporalAtom(a) if index == 0 => Some(Self::Proof(&a.identification, None, None)),
            Self::TemporalAtom(a) if index == 1 => Some(Self::TemporalEnvelope(&a.envelope)),
            Self::Execution(p) => match p {
                RetainedExecutionProof::Admg(c) => c
                    .get(index)
                    .map(|c| c.as_ref().map_or(Self::Absent, |(r, e)| Self::Selected(r, e))),
                RetainedExecutionProof::Pag { envelopes } => envelopes
                    .get(index)
                    .map(|e| e.as_ref().map_or(Self::Absent, Self::TemporalEnvelope)),
                RetainedExecutionProof::CrossWorld { identification, indexer } if index == 0 => {
                    Some(Self::Proof(identification, Some(indexer), None))
                }
                RetainedExecutionProof::CrossWorldClass { envelope } if index == 0 => {
                    Some(Self::TemporalEnvelope(envelope))
                }
                _ => None,
            },
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
    use antecedent_data::TabularData;
    use antecedent_graph::{Dag, DenseNodeId};

    fn prepared() -> PreparedStudy {
        let data = TabularData::from_f64_columns([
            ("t", &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0][..]),
            ("y", &[1.0, 3.1, 1.2, 2.9, 0.9, 3.0, 1.1, 3.2][..]),
        ])
        .unwrap();
        let mut graph = Dag::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        crate::Study::tabular(data)
            .graph(graph)
            .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
            .refute(crate::RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .prepare(&ExecutionContext::for_tests(17))
            .unwrap()
    }

    #[test]
    fn bounded_reads_leave_contract_cache_and_scientific_execution_untouched() {
        let owner = prepared();
        let had_contract = owner.state.contract_cache.get().is_some();
        let observed = antecedent_core::execution_attempt::observe_execution(|| {
            let view = owner.identification_view();
            let summary = view.summary();
            assert_eq!(summary["children"], view.child_count());
            for index in 0..view.child_count() {
                assert!(view.child(index).is_some());
            }
            Ok::<_, crate::CausalError>(summary)
        });
        assert!(observed.result.is_ok());
        assert!(observed.report.is_empty());
        assert_eq!(owner.state.contract_cache.get().is_some(), had_contract);
        let cache = owner.study().static_cache().unwrap();
        let proof = PreparedIdentificationView {
            study: &owner,
            node: Node::Proof(&cache.identification, None, None),
        };
        let expected = serde_json::to_value(
            antecedent_io::identification_to_wire_with_registry(
                &cache.identification,
                owner.population_registry(),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(proof.materialize().unwrap()["identification"], expected);
        assert_eq!(owner.state.contract_cache.get().is_some(), had_contract);
    }

    #[test]
    fn posterior_navigation_preserves_unidentified_samples_and_original_mass() {
        let owner = prepared();
        let graphs = WeightedGraphSamples::new(
            vec![0.2, 0.3, 0.5],
            vec![
                antecedent_prob::GraphIdentFlag::Identified,
                antecedent_prob::GraphIdentFlag::Unidentified,
                antecedent_prob::GraphIdentFlag::Identified,
            ],
            vec![7, 9, 7],
        )
        .unwrap();
        let cache = IdentificationCache::GraphPosterior(Arc::new(
            super::super::CachedGraphPosteriorIdentification {
                graphs,
                atoms: Arc::from([]),
                class_atoms: Arc::from([]),
            },
        ));
        let view = PreparedIdentificationView { study: &owner, node: Node::Cache(&cache) };
        assert_eq!(view.summary()["status"], Value::Null);
        assert_eq!(view.summary()["sample_count"], 3);
        let samples = view.child(0).unwrap();
        assert_eq!(samples.child_count(), 3);
        assert_eq!(samples.child(1).unwrap().summary()["identified"], false);
        assert_eq!(samples.child(1).unwrap().summary()["weight"], 0.3);
        assert_eq!(samples.child(2).unwrap().summary()["key"], 7);
        let converted = samples.materialize().unwrap();
        assert_eq!(converted["children"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn temporal_navigation_preserves_each_horizon_and_its_own_namespace() {
        let owner = prepared();
        let cache = owner.study().static_cache().unwrap();
        let temporal = CachedTemporalIdentification {
            by_horizon: Arc::from([
                CachedTemporalHorizonIdentification {
                    horizon: 1,
                    identification: cache.identification.clone(),
                    estimand: cache.estimand.clone(),
                    indexer: antecedent_core::TemporalIndexer::new(2, 1, 1).unwrap(),
                },
                CachedTemporalHorizonIdentification {
                    horizon: 3,
                    identification: cache.identification.clone(),
                    estimand: cache.estimand.clone(),
                    indexer: antecedent_core::TemporalIndexer::new(2, 2, 3).unwrap(),
                },
            ]),
        };
        let view = PreparedIdentificationView { study: &owner, node: Node::Temporal(&temporal) };
        assert_eq!(view.child_count(), 2);
        assert_eq!(view.child(1).unwrap().summary()["horizon"], 3);
        let last = view.child(1).unwrap().child(0).unwrap().summary();
        assert_eq!(last["temporal_namespace"]["history"], 2);
        assert_eq!(last["temporal_namespace"]["horizon"], 3);
    }

    #[test]
    fn absent_first_component_does_not_hide_later_proofs_and_cancel_stops_materialization() {
        let owner = prepared();
        let cache = owner.study().static_cache().unwrap();
        let proof = RetainedExecutionProof::Admg(vec![
            None,
            Some((cache.identification.clone(), cache.estimand.clone())),
        ]);
        let view = PreparedIdentificationView { study: &owner, node: Node::Execution(&proof) };
        assert_eq!(view.child_count(), 2);
        assert_eq!(view.child(0).unwrap().summary()["kind"], "absent_component");
        assert_eq!(view.child(1).unwrap().child(0).unwrap().summary()["kind"], "proof");
        let converted = view.materialize().unwrap();
        assert_eq!(converted["children"].as_array().unwrap().len(), 2);
        let ctx = ExecutionContext::for_tests(17);
        ctx.cancellation.cancel();
        assert!(matches!(
            view.materialize_with_context(&ctx),
            Err(crate::CausalError::Cancelled { .. })
        ));
    }
    fn named_prepared() -> PreparedStudy {
        let z = [0., 0., 1., 1., 0., 0., 1., 1.];
        let t = [0., 1., 0., 1., 0., 1., 0., 1.];
        let y = [0., 2., 1., 3., 0.1, 1.9, 1.1, 2.9];
        let data =
            TabularData::from_f64_columns([("z", &z[..]), ("t", &t[..]), ("y", &y[..])]).unwrap();
        let mut graph = Dag::with_variables(3);
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        crate::Study::tabular(data)
            .graph(graph)
            .query(AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2)))
            .refute(crate::RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .prepare_deferred(&ExecutionContext::for_tests(17))
            .unwrap()
    }

    #[test]
    fn selected_proof_pages_read_real_adjustment_and_original_assumptions_without_execution() {
        let owner = named_prepared();
        let cache = owner.study().static_cache().unwrap();
        let view = PreparedIdentificationView {
            study: &owner,
            node: Node::Proof(&cache.identification, None, None),
        };
        assert!(owner.identification_view().estimands_page(0, 32).is_err());
        let observed = antecedent_core::execution_attempt::observe_execution(|| {
            let estimands = view.estimands_page(0, 32)?;
            assert_eq!(estimands["total"], cache.identification.estimands.len());
            let adjustment = view.adjustment_page(Some(0), 0, 32)?;
            assert_eq!(adjustment["items"], json!([{"name":"z","variable":0,"offset":0,"node":0}]));
            let assumptions = view.assumptions_page(None, 0, 256)?;
            assert_eq!(
                assumptions["items"],
                serde_json::to_value(antecedent_io::assumptions_to_wire(
                    &cache.identification.required_assumptions
                ))
                .unwrap()
            );
            let steps = view.derivation_page(0, 256)?;
            assert_eq!(steps["total"], cache.identification.derivation.steps.len());
            for (item, step) in steps["items"]
                .as_array()
                .unwrap()
                .iter()
                .zip(&cache.identification.derivation.steps)
            {
                assert_eq!(item["rule"], step.rule.as_ref());
                assert_eq!(item["detail"], step.detail.as_ref());
            }
            Ok::<_, crate::CausalError>(())
        });
        assert!(observed.result.is_ok());
        assert!(observed.report.is_empty());
    }

    #[test]
    fn coordinate_pages_use_the_retained_temporal_indexer_and_never_dense_schema_guessing() {
        // Coordinate projection fixture, not a new temporal identification theorem.
        let owner = named_prepared();
        let cache = owner.study().static_cache().unwrap();
        let mut proof = cache.identification.clone();
        let indexer = antecedent_core::TemporalIndexer::new(3, 2, 3).unwrap();
        let key =
            antecedent_core::TemporalNodeKey { variable: VariableId::from_raw(0), offset: -1 };
        let dense = indexer.dense_id(key).unwrap();
        proof.estimands[0].method = Arc::from("temporal.backdoor.unfolded");
        proof.estimands[0].adjustment_set = Arc::from([VariableId::from_raw(dense)]);
        let indexed = PreparedIdentificationView {
            study: &owner,
            node: Node::Proof(&proof, Some(&indexer), None),
        };
        assert_eq!(
            indexed.adjustment_page(Some(0), 0, 1).unwrap()["items"],
            json!([{"name":"z","variable":0,"offset":-1,"node":dense}])
        );
        let absent =
            PreparedIdentificationView { study: &owner, node: Node::Proof(&proof, None, None) };
        assert_eq!(
            absent.adjustment_page(Some(0), 0, 1).unwrap()["reason"],
            "temporal_namespace_not_retained"
        );
    }

    #[test]
    fn alternative_estimands_require_selection_and_empty_claim_vector_has_native_shared_semantics()
    {
        let owner = named_prepared();
        let cache = owner.study().static_cache().unwrap();
        let mut proof = cache.identification.clone();
        proof.estimands.push(proof.estimands[0].clone());
        proof.estimand_claims.clear();
        let view =
            PreparedIdentificationView { study: &owner, node: Node::Proof(&proof, None, None) };
        assert!(view.adjustment_page(None, 0, 32).is_err());
        assert!(view.adjustment_page(Some(5), 0, 32).is_err());
        assert_eq!(
            view.assumptions_page(Some(1), 0, 32).unwrap()["scope"],
            "estimand_shared_native_proof"
        );
        assert!(view.derivation_page(0, 257).is_err());
        assert!(view.estimands_page(0, 257).is_err());
        assert!(view.assumptions_page(None, 0, 257).is_err());
        assert!(view.adjustment_page(Some(0), 0, 257).is_err());
        assert!(view.estimands_page(20, 1).unwrap()["items"].as_array().unwrap().is_empty());
    }

    #[test]
    fn transport_node_summary_keeps_the_native_actionable_outcome_domain() {
        let owner = prepared();
        let proof = antecedent_identify::TransportIdentification::MissingEvidence(
            antecedent_identify::MissingEvidenceCertificate {
                reason: Arc::from("required_source_regime_missing"),
                missing: Arc::from([VariableId::from_raw(0)]),
                message: Arc::from("caller must supply declared source evidence"),
            },
        );
        let view = PreparedIdentificationView { study: &owner, node: Node::Transport(&proof) };
        let summary = view.summary();
        assert_eq!(summary["status"], "missing_evidence");
        assert_eq!(summary["status_domain"], "transport_outcome");
        assert_eq!(summary["reason"], "required_source_regime_missing");
        assert_eq!(summary["location_node_count"], 1);
        assert!(summary["status_unavailable"].is_null());
    }
}
