//! Original scalar effect coordinates for the public result-to-decision handoff.
//!
//! A contrast is one scientific quantity, never two reconstructed outcome means.

use antecedent::StudyResult;
use antecedent::analysis::decision_result::{
    DecisionProjectionError, NativeEffectAuthority as EffectAuthority, project_static_effect,
};
use antecedent::io::quantity_wire::ScientificQuantityWire;
use pyo3::prelude::*;

use crate::bridge::external_refusal_json;

#[derive(Clone, Debug)]
pub(crate) struct NativeEffectAuthority {
    inner: EffectAuthority,
    names: Vec<String>,
}

impl NativeEffectAuthority {
    pub(crate) fn from_result(result: &StudyResult, names: &[String]) -> Option<Self> {
        Some(Self { inner: EffectAuthority::from_result(result)?, names: names.to_vec() })
    }
}

#[pymethods]
impl crate::AteAnalysisResult {
    /// Read a checked original static ATE as an affine mean input. Declared units
    /// label the original outcome scale; no conversion or outcome-law claim occurs.
    fn effect_source_json(
        &self,
        units: &str,
        population: &str,
        supplied_value: f64,
    ) -> PyResult<(Option<String>, Option<String>)> {
        crate::catch_ffi(|| {
            let authority = self.effect_authority.as_ref();
            let projected = match project_static_effect(
                authority.map(|authority| &authority.inner),
                units,
                population,
                supplied_value,
                |id| authority?.names.get(id.raw() as usize).cloned(),
            ) {
                Ok(projected) => projected,
                Err(DecisionProjectionError::Declaration(message)) => {
                    return Err(crate::value_err(message));
                }
                Err(DecisionProjectionError::Refusal(refusal)) => {
                    return Ok((None, Some(external_refusal_json(&refusal))));
                }
            };
            let source = projected.source;
            let coordinates =
                source.coordinates.iter().map(ScientificQuantityWire::from).collect::<Vec<_>>();
            let wire = serde_json::json!({
                "coordinates": coordinates,
                "means": source.means,
                "provider_id": source.provider_id,
                "snapshot_id": source.snapshot_id,
                "causal_contract_id": source.causal_contract_id,
                "query": {
                    "treatment": projected.treatment,
                    "outcome": projected.outcome,
                    "control": projected.control,
                    "active": projected.active,
                    "adjustment": self.adjustment_set,
                },
                "rng_id": source.rng_id,
            });
            Ok((Some(wire.to_string()), None))
        })
    }
}
