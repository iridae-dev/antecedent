//! Immutable prepared metadata, captured before publication and borrowed on inspection.
use super::*;
use serde_json::{Value, json};

/// Exact identities and native reasoning retained at a preparation boundary.
/// Complete proof arenas and numerical replay payloads are not part of this projection.
#[derive(Clone, Debug)]
pub struct PreparedInspection {
    target: antecedent_core::SemanticDigest,
    identification: antecedent_core::SemanticDigest,
    identification_product: Option<antecedent_core::SemanticDigest>,
    program: antecedent_core::SemanticDigest,
    inference_binding: antecedent_core::SemanticDigest,
    observation: antecedent_core::SemanticDigest,
    snapshot: Option<antecedent_core::SemanticDigest>,
    reasoning: ReasoningView,
    evidence_obligations: Arc<[antecedent_contract::EvidenceObligation]>,
    evidence_actions: Arc<[NextAction]>,
    graph_class: GraphClass,
    structure_source: StructureSource,
    accepted_version: u32,
    identifier: Option<Arc<str>>,
    estimator: Option<Arc<str>>,
    modality: Arc<str>,
    inference: Arc<str>,
}

impl PreparedStudy {
    /// Capture the data-independent identity pipeline and native reasoning.
    /// This must run at a completed preparation boundary, never while rendering.
    pub(crate) fn seal_inspection_metadata(&self) -> Result<(), CausalError> {
        if self.inspection_header().get().is_some() {
            return Ok(());
        }
        let program = self.program_payloads()?;
        let cached = contract_identification(self.study());
        let proof = cached.as_deref();
        let search_capped = proof.is_some_and(identification_search_capped);
        let study = self.study();
        let resolved = self.plan().logical.record.estimator.as_deref();
        let (evidence_obligations, evidence_actions) = evidence_needs::derive(study, proof);
        let header = PreparedInspection {
            target: program.target_digest,
            identification: program.identification_digest,
            identification_product: program.identification_product_digest,
            program: program.program_digest,
            inference_binding: program.inference_binding_digest,
            observation: program.observation_digest,
            snapshot: self.contract_cache().get().map(|c| c.contract.identities.data_snapshot),
            reasoning: reasoning_view(study, Some(self), proof, search_capped),
            evidence_obligations,
            evidence_actions,
            graph_class: study.graph.class(),
            structure_source: study.structure_source,
            accepted_version: study.graph.version(),
            identifier: study.identifier.map(|id| Arc::from(id.as_str())),
            estimator: advertised_estimator(study, resolved).map(Arc::from),
            modality: Arc::from(snapshot_modality(&study.data)),
            inference: Arc::from(declared_inference(study)),
        };
        let _ = self.inspection_header().set(Arc::new(header));
        Ok(())
    }

    /// Borrow the already captured prepared projection. Missing metadata is an
    /// implementation error in the preparation route, never a structural fallback.
    ///
    /// # Errors
    /// The preparation route failed to seal metadata before publication.
    pub fn prepared_inspection(&self) -> Result<&PreparedInspection, CausalError> {
        self.inspection_header().get().map(Arc::as_ref).ok_or_else(|| CausalError::Compile {
            message: "prepared inspection metadata was not sealed".into(),
        })
    }
}

fn availability<T>(slot: &SlotAvailability<T>) -> Value {
    match slot {
        SlotAvailability::Available(_) => json!({"available":true}),
        SlotAvailability::Unavailable { reason } => json!({"available":false,"reason":reason}),
        _ => json!({"available":false,"reason":"unknown_native_slot"}),
    }
}

fn scope_coordinate(scope: ObligationScope) -> Value {
    match scope {
        ObligationScope::Horizon { horizon } => json!({"kind":"horizon","horizon":horizon}),
        _ => json!({"kind":scope.as_str()}),
    }
}

fn assumption_source(source: &AssumptionSource) -> Value {
    match source {
        AssumptionSource::UserDeclared => json!({"kind":"user_declared"}),
        AssumptionSource::AlgorithmDefault { algorithm } => {
            json!({"kind":"algorithm_default","algorithm":algorithm})
        }
        AssumptionSource::Artifact => json!({"kind":"artifact"}),
        AssumptionSource::Derived { from } => json!({"kind":"derived","from":from}),
    }
}

fn page_end(length: usize, offset: usize, limit: usize) -> Result<usize, CausalError> {
    if limit > 256 {
        return Err(CausalError::Compile {
            message: "inspection page limit must be at most 256".into(),
        });
    }
    Ok(offset.saturating_add(limit).min(length))
}

impl PreparedInspection {
    /// Borrow the complete native slots without converting any payload; hosts read the
    /// bounded summary and pages instead.
    #[cfg(test)]
    #[must_use]
    pub(crate) const fn reasoning(&self) -> &ReasoningView {
        &self.reasoning
    }

    /// Borrow all original evidence obligations; host adapters page this slice.
    #[must_use]
    pub fn evidence_obligations(&self) -> &[antecedent_contract::EvidenceObligation] {
        &self.evidence_obligations
    }

    /// Borrow original evidence-gathering actions in native order.
    #[must_use]
    pub fn evidence_actions(&self) -> &[NextAction] {
        &self.evidence_actions
    }

    /// Fixed-size identity and four-slot metadata. Variable-length obligations
    /// and uncertainty components are available only through explicit pages.
    #[must_use]
    pub fn summary(&self) -> Value {
        let mut identification = availability(&self.reasoning.identification);
        if let Some(slot) = self.reasoning.identification.as_ref() {
            identification["status"] = json!(slot.status.as_str());
            identification["identified_mass"] = json!(slot.identified_mass);
            identification["unidentified_mass"] = json!(slot.unidentified_mass);
            identification["unevaluable_mass"] = json!(slot.unevaluable_mass);
            identification["incomplete_search_mass"] = json!(slot.incomplete_search_mass);
            identification["full_mass_scope"] = json!(slot.full_mass_scope);
            identification["search_capped"] = json!(slot.search_capped);
            identification["weight_basis"] = json!(slot.weight_basis);
        }
        let mut support = availability(&self.reasoning.support);
        if let Some(slot) = self.reasoning.support.as_ref() {
            support["matrix_status"] = json!(slot.matrix_status);
            support["matrix_coordinate"] = json!(slot.matrix_coordinate);
            support["empirical"] = availability(&slot.empirical);
            if let Some(empirical) = slot.empirical.as_ref() {
                support["empirical"]["status"] = json!(empirical);
            }
        }
        let mut uncertainty = availability(&self.reasoning.uncertainty);
        uncertainty["component_count"] =
            json!(self.reasoning.uncertainty.as_ref().map_or(0, |s| s.components.len()));
        let mut assumptions = availability(&self.reasoning.assumptions);
        assumptions["obligation_count"] =
            json!(self.reasoning.assumptions.as_ref().map_or(0, |s| s.obligations.len()));
        json!({
            "stage":"prepared",
            "evidence_obligation_count":self.evidence_obligations.len(),
            "evidence_action_count":self.evidence_actions.len(),
            "identities":{
                "target_id":self.target.to_hex(),
                "identification_id":self.identification.to_hex(),
                "identification_product_id":self.identification_product.map(|d|d.to_hex()),
                "program_id":self.program.to_hex(),
                "inference_binding_id":self.inference_binding.to_hex(),
                "observation_id":self.observation.to_hex(),
                "data_snapshot_id":self.snapshot.map(|d|d.to_hex()),
                "snapshot_unavailable":self.snapshot.is_none().then_some("prepared_snapshot_identity_not_materialized"),
            },
            "identification":identification,"support":support,
            "uncertainty":uncertainty,"assumptions":assumptions,
            "graph_class":self.graph_class.as_str(),
            "structure_source":self.structure_source.as_str(),
            "accepted_version":self.accepted_version,
            "identifier":self.identifier,"estimator":self.estimator,
            "modality":self.modality,"inference":self.inference,
        })
    }

    /// Original scoped obligations in retained order; no aggregate count or
    /// truncated list is presented as the complete assumption set.
    ///
    /// # Errors
    /// A page requests more than 256 entries.
    pub fn obligations_page(&self, offset: usize, limit: usize) -> Result<Value, CausalError> {
        let obligations =
            self.reasoning.assumptions.as_ref().map(|s| s.obligations.as_ref()).unwrap_or(&[]);
        let end = page_end(obligations.len(), offset, limit)?;
        let items = obligations.get(offset..end).unwrap_or(&[]).iter().map(|o|json!({
            "id":o.id,"scope":o.scope.as_str(),"scope_coordinate":scope_coordinate(o.scope),
            "source":assumption_source(&o.source),"kind":o.kind.as_str(),"status":o.status.as_str(),
            "required_check":o.required_check,"evidence":o.evidence,
            "assumption_display":format!("{:?}",o.assumption),"description":o.description,
        })).collect::<Vec<_>>();
        Ok(json!({"offset":offset,"total":obligations.len(),"items":items,
            "next_offset":(end<obligations.len()).then_some(end)}))
    }

    /// Original uncertainty components in retained order.
    ///
    /// # Errors
    /// A page requests more than 256 entries.
    pub fn uncertainty_page(&self, offset: usize, limit: usize) -> Result<Value, CausalError> {
        let components =
            self.reasoning.uncertainty.as_ref().map(|s| s.components.as_ref()).unwrap_or(&[]);
        let end = page_end(components.len(), offset, limit)?;
        let items = components
            .get(offset..end)
            .unwrap_or(&[])
            .iter()
            .map(|c| {
                json!({
                    "source":c.source.as_str(),"target":c.target,"omitted":c.omitted,
                })
            })
            .collect::<Vec<_>>();
        Ok(json!({"offset":offset,"total":components.len(),"items":items,
            "next_offset":(end<components.len()).then_some(end)}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::execution_attempt::{Operation, observe_execution};
    use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
    use antecedent_data::TabularData;
    use antecedent_graph::{Dag, DenseNodeId};

    fn data(shift: f64) -> TabularData {
        TabularData::from_f64_columns([
            ("z", &[0., 0., 1., 1., 0., 0., 1., 1.][..]),
            ("t", &[0., 1., 0., 1., 0., 1., 0., 1.][..]),
            (
                "y",
                &[
                    shift,
                    2. + shift,
                    1. + shift,
                    3. + shift,
                    0.1 + shift,
                    1.9 + shift,
                    1.1 + shift,
                    2.9 + shift,
                ][..],
            ),
        ])
        .unwrap()
    }

    fn study() -> Study {
        let mut graph = Dag::with_variables(3);
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        crate::Study::tabular(data(0.))
            .graph(graph)
            .query(AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2)))
            .refute(crate::RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
    }

    #[test]
    fn preparing_and_reading_metadata_never_executes_numerical_work() {
        let observed = observe_execution(|| {
            let ready = study().prepare_deferred(&ExecutionContext::for_tests(17))?;
            let before = ready.contract_cache().get().is_some();
            for _ in 0..4 {
                let header = ready.prepared_inspection()?;
                let summary = header.summary();
                assert_eq!(summary["stage"], "prepared");
                assert_eq!(summary["identification"]["available"], true);
                assert_eq!(summary["uncertainty"]["available"], false);
                assert_eq!(summary["uncertainty"]["reason"], "execution_specific");
                assert_eq!(
                    summary["identities"]["snapshot_unavailable"],
                    "prepared_snapshot_identity_not_materialized"
                );
                assert_eq!(
                    header.obligations_page(0, 32)?["total"],
                    summary["assumptions"]["obligation_count"]
                );
                assert_eq!(header.uncertainty_page(0, 32)?["total"], 0);
                let proof = ready.identification_view();
                assert_eq!(proof.summary()["children"], proof.child_count());
                assert_eq!(ready.contract_cache().get().is_some(), before);
            }
            Ok::<_, CausalError>(ready)
        });
        assert!(observed.result.is_ok());
        for operation in [
            Operation::LeastSquaresSolve,
            Operation::GlmFit,
            Operation::AdjustedFit,
            Operation::LearnerFit,
            Operation::PosteriorFit,
            Operation::PosteriorDraw,
            Operation::ScoreConstruction,
            Operation::FactorConstruction,
            Operation::FactorEvaluation,
            Operation::Integration,
            Operation::ProviderInvocation,
            Operation::LawSummary,
            Operation::Decision,
        ] {
            assert_eq!(observed.report.counts(operation).attempted, 0, "{operation:?}");
        }
    }

    #[test]
    fn prepared_domains_equal_original_contract_without_promoting_structural_or_storage_ids() {
        let ready = study().prepare_deferred(&ExecutionContext::for_tests(17)).unwrap();
        let summary = ready.prepared_inspection().unwrap().summary();
        let actual = ready.contract().unwrap();
        let identities = &summary["identities"];
        assert_eq!(identities["target_id"], actual.identities.target.to_hex());
        assert_eq!(identities["identification_id"], actual.identities.identification.to_hex());
        assert_eq!(
            identities["identification_product_id"],
            json!(actual.identities.identification_product.map(|d| d.to_hex()))
        );
        assert_eq!(identities["program_id"], json!(actual.identities.program.map(|d| d.to_hex())));
        assert_eq!(
            identities["inference_binding_id"],
            actual.identities.inference_binding.to_hex()
        );
        assert_eq!(identities["observation_id"], actual.identities.observation.to_hex());
        assert_eq!(ready.prepared_inspection().unwrap().reasoning(), &actual.reasoning);
        assert!(identities["data_snapshot_id"].is_null());
        assert_eq!(
            ready.prepared_inspection().unwrap().summary(),
            summary,
            "creating a later contract must not rewrite the prepared record"
        );
    }

    #[test]
    fn rebound_and_requoted_states_seal_their_own_metadata_without_mutating_parent() {
        let ready = study().prepare_deferred(&ExecutionContext::for_tests(17)).unwrap();
        let original = ready.prepared_inspection().unwrap().summary();
        let branch = ready.rebind_data(data(10.), &ExecutionContext::for_tests(17)).unwrap();
        let requoted = branch.reconfigure_execution(&ExecutionContext::for_tests(29)).unwrap();
        assert_eq!(ready.prepared_inspection().unwrap().summary(), original);
        for child in [&branch, &requoted] {
            let summary = child.prepared_inspection().unwrap().summary();
            let contract = child.contract().unwrap();
            assert_eq!(
                summary["identities"]["program_id"],
                json!(contract.identities.program.map(|d| d.to_hex()))
            );
            assert!(summary["identities"]["data_snapshot_id"].is_null());
            assert_eq!(summary["identities"]["target_id"], original["identities"]["target_id"]);
        }
    }

    #[test]
    fn obligation_pages_keep_all_original_ordered_scopes_without_a_truncated_complete_claim() {
        let ready = study().prepare_deferred(&ExecutionContext::for_tests(17)).unwrap();
        let header = ready.prepared_inspection().unwrap();
        let obligations = &header.reasoning.assumptions.as_ref().unwrap().obligations;
        let mut collected = Vec::new();
        let mut offset = 0;
        while offset < obligations.len() {
            let page = header.obligations_page(offset, 1).unwrap();
            assert_eq!(page["total"], obligations.len());
            collected.push(page["items"][0]["id"].as_str().unwrap().to_string());
            offset += 1;
        }
        assert_eq!(collected, obligations.iter().map(|o| o.id.to_string()).collect::<Vec<_>>());
        assert!(header.obligations_page(0, 257).is_err());
        assert!(
            header.obligations_page(obligations.len() + 1, 32).unwrap()["items"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    }
}
