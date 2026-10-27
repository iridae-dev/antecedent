//! Retained trial proof navigation. Reading never reidentifies, fits, or scans rows.
use super::PreparedLearnedTrial;
use antecedent::{LearnedTrialState, PreparedStudy};
use antecedent_identify::{TransportCertificate, TransportFormula, TransportIdentification};
use pyo3::{prelude::*, types::PyString};
use serde_json::{Value, json};
use std::sync::Arc;

#[pyclass(name = "TrialIdentificationView", frozen)]
struct TrialIdentificationView {
    owner: Arc<PreparedStudy<LearnedTrialState>>,
    names: Arc<[String]>,
}

fn end(total: usize, offset: usize, limit: usize) -> PyResult<usize> {
    if limit > 256 {
        return Err(crate::value_err("identification page limit must be at most 256"));
    }
    Ok(offset.saturating_add(limit).min(total))
}
fn page(total: usize, offset: usize, end: usize, items: Vec<Value>) -> Value {
    json!({"available":true,"total":total,"offset":offset,"items":items,
        "next_offset":(end<total).then_some(end)})
}
fn unavailable(offset: usize, reason: &str) -> Value {
    json!({"available":false,"reason":reason,"total":null,"offset":offset,
        "items":[],"next_offset":null})
}
fn encode(value: &Value) -> PyResult<String> {
    serde_json::to_string(value).map_err(|error| crate::value_err(error.to_string()))
}
fn formula_kind(formula: &TransportFormula) -> &'static str {
    match formula {
        TransportFormula::Direct(_) => "direct",
        TransportFormula::Standardize { .. } => "standardize",
        TransportFormula::RecursiveFactorization { .. } => "recursive_factorization",
    }
}

impl TrialIdentificationView {
    fn proof(&self) -> &TransportIdentification {
        self.owner.identification()
    }
    fn positive(&self) -> Option<(&TransportFormula, &TransportCertificate)> {
        match self.proof() {
            TransportIdentification::Transportable { formula, certificate } => {
                Some((formula, certificate))
            }
            _ => None,
        }
    }
    fn summary(&self) -> Value {
        let outcome = self.proof().outcome();
        let mut value = json!({"kind":"transport","children":0,"status":outcome.kind.as_str(),
            "status_domain":"transport_outcome","status_unavailable":null,
            "reason":outcome.reason,"method":null});
        if let Some((formula, certificate)) = self.positive() {
            value["method"] = json!(certificate.rule);
            value["formula_kind"] = json!(formula_kind(formula));
            value["premise_count"] = json!(certificate.premises.len());
            value["selection_count"] = json!(certificate.selection_targets.len());
        }
        value
    }
    fn selection(&self, estimand: Option<usize>) -> PyResult<()> {
        if estimand.is_some_and(|index| index != 0) {
            return Err(crate::value_err(
                "transport identification contains one formula; index must be 0",
            ));
        }
        Ok(())
    }
}

#[pymethods]
impl TrialIdentificationView {
    fn summary_json(&self) -> PyResult<String> {
        encode(&self.summary())
    }
    fn child(&self, _index: usize) -> PyResult<Self> {
        Err(crate::value_err("a retained trial transport proof has no child proofs"))
    }
    #[pyo3(signature=(*,offset=0,limit=32))]
    fn children(&self, offset: usize, limit: usize) -> PyResult<Vec<Self>> {
        end(0, offset, limit)?;
        Ok(vec![])
    }
    #[pyo3(signature=(*,offset=0,limit=32))]
    fn estimands_json(&self, offset: usize, limit: usize) -> PyResult<String> {
        let Some((formula, certificate)) = self.positive() else {
            end(0, offset, limit)?;
            return encode(&unavailable(offset, "no_certified_transport_formula"));
        };
        let stop = end(1, offset, limit)?;
        let items = if offset < stop {
            vec![json!({"index":0,"kind":"transport_formula",
            "method":certificate.rule,"formula_kind":formula_kind(formula),
            "meaning":"population_labelled_transport_formula","functional_arena":null})]
        } else {
            vec![]
        };
        encode(&page(1, offset, stop, items))
    }
    #[pyo3(signature=(*,estimand=None,offset=0,limit=32))]
    fn adjustment_json(
        &self,
        estimand: Option<usize>,
        offset: usize,
        limit: usize,
    ) -> PyResult<String> {
        self.selection(estimand)?;
        end(0, offset, limit)?;
        let coordinates = match self.positive().map(|(formula, _)| formula) {
            Some(TransportFormula::Direct(_)) => &[][..],
            Some(TransportFormula::Standardize { over, .. }) => over.as_ref(),
            Some(TransportFormula::RecursiveFactorization { .. }) => {
                return encode(&unavailable(
                    offset,
                    "recursive_transport_formula_has_no_single_adjustment_set",
                ));
            }
            None => return encode(&unavailable(offset, "no_certified_transport_formula")),
        };
        let stop = end(coordinates.len(), offset, limit)?;
        let items = coordinates
            .get(offset..stop)
            .unwrap_or(&[])
            .iter()
            .map(|id| {
                let name = self.names.get(id.as_usize()).ok_or_else(|| {
                    crate::value_err("retained transport coordinate is outside its schema")
                })?;
                Ok(json!({"variable":id.raw(),"node":id.raw(),"offset":0,"name":name}))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let mut result = page(coordinates.len(), offset, stop, items);
        result["meaning"] = json!("certified_transport_standardizers");
        encode(&result)
    }
    #[pyo3(signature=(*,estimand=None,offset=0,limit=32))]
    fn assumptions_json(
        &self,
        estimand: Option<usize>,
        offset: usize,
        limit: usize,
    ) -> PyResult<String> {
        self.selection(estimand)?;
        end(0, offset, limit)?;
        let Some((_, certificate)) = self.positive() else {
            return encode(&unavailable(offset, "no_certified_transport_formula"));
        };
        let stop = end(certificate.premises.len(), offset, limit)?;
        let items = certificate
            .premises
            .get(offset..stop)
            .unwrap_or(&[])
            .iter()
            .enumerate()
            .map(|(i, premise)| {
                json!({"index":offset+i,"premise":premise,
                "rule":certificate.rule,"scope":"transport_certificate"})
            })
            .collect();
        let mut result = page(certificate.premises.len(), offset, stop, items);
        result["meaning"] = json!("checked_native_transport_certificate_premises");
        encode(&result)
    }
    #[pyo3(signature=(*,offset=0,limit=32))]
    fn derivation_json(&self, offset: usize, limit: usize) -> PyResult<String> {
        // The actual retained object has certificate premises, not an expression-arena log.
        let encoded = self.assumptions_json(None, offset, limit)?;
        Ok(encoded)
    }
    #[pyo3(signature=(*,seed=1,threads=None))]
    fn materialize_json(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<Py<PyString>> {
        let context = crate::py_execution_context(seed, crate::resolve_user_threads(threads));
        let summary = self.summary();
        let payload = materialize::Payload {
            summary: &summary,
            transport: materialize::Proof::from(self.proof()),
            children: [],
        };
        let encoded = materialize::admitted(&payload, &context)?;
        let text = encoded.as_str()?;
        // Explicit Python conversion is the ownership/materialization boundary.
        Ok(PyString::new(py, text).unbind())
    }
    fn schema_names(&self) -> Vec<String> {
        self.names.to_vec()
    }
    fn __repr__(&self) -> String {
        format!("TrialIdentificationView(status={})", self.proof().outcome().kind.as_str())
    }
}

#[pymethods]
impl PreparedLearnedTrial {
    fn identification_view(&self) -> PyResult<TrialIdentificationView> {
        let owner = self.inner.as_ref().ok_or_else(|| {
            crate::refusal(
                "route_not_supported",
                "loaded trial artifacts do not retain a prepared structural proof",
            )
        })?;
        Ok(TrialIdentificationView { owner: Arc::clone(owner), names: Arc::clone(&self.names) })
    }
    #[pyo3(signature=(*,offset=0,limit=32))]
    fn prepared_obligations_json(&self, offset: usize, limit: usize) -> PyResult<String> {
        end(0, offset, limit)?;
        let Some(reasoning) = self.reasoning() else {
            return encode(&unavailable(offset, "reasoning_not_retained"));
        };
        let Some(slot) = reasoning.assumptions.as_ref() else {
            return encode(&unavailable(offset, "assumptions_not_retained"));
        };
        let stop = end(slot.obligations.len(), offset, limit)?;
        let items = slot
            .obligations
            .get(offset..stop)
            .unwrap_or(&[])
            .iter()
            .map(|o| {
                let source = match &o.source {
                    antecedent_core::AssumptionSource::UserDeclared => {
                        json!({"kind":"user_declared"})
                    }
                    antecedent_core::AssumptionSource::AlgorithmDefault { algorithm } => {
                        json!({"kind":"algorithm_default","algorithm":algorithm})
                    }
                    antecedent_core::AssumptionSource::Artifact => json!({"kind":"artifact"}),
                    antecedent_core::AssumptionSource::Derived { from } => {
                        json!({"kind":"derived","from":from})
                    }
                };
                json!({"id":o.id,"scope":o.scope.as_str(),"source":source,"kind":o.kind.as_str(),
                "status":o.status.as_str(),"required_check":o.required_check,"evidence":o.evidence,
                "assumption_display":format!("{:?}",o.assumption),"description":o.description})
            })
            .collect();
        encode(&page(slot.obligations.len(), offset, stop, items))
    }
    #[pyo3(signature=(*,offset=0,limit=32))]
    fn prepared_uncertainty_json(&self, offset: usize, limit: usize) -> PyResult<String> {
        end(0, offset, limit)?;
        let Some(reasoning) = self.reasoning() else {
            return encode(&unavailable(offset, "reasoning_not_retained"));
        };
        let Some(slot) = reasoning.uncertainty.as_ref() else {
            let reason = match &reasoning.uncertainty {
                antecedent_contract::SlotAvailability::Unavailable { reason } => reason.as_ref(),
                _ => unreachable!("checked slot above"),
            };
            return encode(&unavailable(offset, reason));
        };
        let stop = end(slot.components.len(), offset, limit)?;
        let items = slot
            .components
            .get(offset..stop)
            .unwrap_or(&[])
            .iter()
            .map(|c| json!({"source":c.source.as_str(),"target":c.target,"omitted":c.omitted}))
            .collect();
        encode(&page(slot.components.len(), offset, stop, items))
    }
    #[pyo3(signature=(*,offset=0,limit=32))]
    fn prepared_evidence_obligations_json(&self, offset: usize, limit: usize) -> PyResult<String> {
        end(0, offset, limit)?;
        // This native modality retains graphical premises and sampling declarations,
        // not an EvidenceObligation ledger. Never label an absent ledger empty/passed.
        encode(&unavailable(offset, "trial_evidence_obligation_ledger_not_retained"))
    }
    #[pyo3(signature=(*,offset=0,limit=32))]
    fn prepared_evidence_actions_json(&self, offset: usize, limit: usize) -> PyResult<String> {
        end(0, offset, limit)?;
        encode(&unavailable(offset, "trial_evidence_action_ledger_not_retained"))
    }
}

#[path = "learned_trial_readmodel_materialize.rs"]
mod materialize;

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<TrialIdentificationView>()
}
