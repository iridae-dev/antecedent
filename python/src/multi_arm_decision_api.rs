//! Arm means of a multi-arm randomized result, read back as decision coordinates.
//!
//! The scalar point stays the second-arm contrast. This method reads the retained arm means.

use antecedent::StudyResult;
use antecedent::analysis::decision_result::{
    DecisionProjectionError, MULTI_ARM_PROVIDER_ID, MULTI_ARM_RNG_ID, MultiArmDecisionAuthority,
    MultiArmMeanProjection, project_multi_arm_means,
};
use antecedent::io::quantity_wire::ScientificQuantityWire;
use pyo3::prelude::*;

use crate::bridge::external_refusal_json;

const OBJECT_OF_STRINGS: &str = "units and regimes are objects of strings";

#[derive(Clone, Debug)]
pub(crate) struct MultiArmAuthority {
    inner: MultiArmDecisionAuthority,
    outcome: Option<String>,
}

impl MultiArmAuthority {
    pub(crate) fn from_result(result: &StudyResult, names: &[String]) -> Option<Self> {
        let inner = MultiArmDecisionAuthority::from_result(result)?;
        let outcome = names.get(inner.outcome().raw() as usize).cloned();
        Some(Self { inner, outcome })
    }
}

#[pymethods]
impl crate::AteAnalysisResult {
    /// One mean coordinate per arm of this multi-arm result.
    ///
    /// `units_json` and `regimes_json` are objects of strings. Units are keyed by the
    /// outcome's name; regimes map each arm label to the regime the decision uses.
    /// Variances are not returned. A result that is not a multi-arm experiment refuses.
    fn multi_arm_means_json(
        &self,
        population: &str,
        units_json: &str,
        regimes_json: &str,
    ) -> PyResult<(Option<String>, Option<String>)> {
        crate::catch_ffi(|| {
            let stored = self.multi_arm_authority.as_ref();
            let prepared = match stored {
                Some(stored) => match prepare(stored, units_json, regimes_json) {
                    Ok(prepared) => Some(prepared),
                    Err(error) => return refused(error),
                },
                None => None,
            };
            let (units, regimes, outcome) =
                prepared.unwrap_or_else(|| (String::new(), Vec::new(), String::new()));
            let regime_refs: Vec<(&str, &str)> =
                regimes.iter().map(|(label, regime)| (label.as_str(), regime.as_str())).collect();
            match project_multi_arm_means(
                stored.map(|stored| &stored.inner),
                population,
                &units,
                &regime_refs,
                &outcome,
            ) {
                Ok(projected) => Ok((Some(projection_json(&projected)), None)),
                Err(error) => refused(error),
            }
        })
    }
}

fn prepare(
    stored: &MultiArmAuthority,
    units_json: &str,
    regimes_json: &str,
) -> Result<(String, Vec<(String, String)>, String), DecisionProjectionError> {
    let outcome = stored.outcome.clone().unwrap_or_default();
    let units = string_object(units_json)?
        .into_iter()
        .find(|(key, _)| key == &outcome)
        .map(|(_, value)| value)
        .unwrap_or_default();
    Ok((units, string_object(regimes_json)?, outcome))
}

fn string_object(text: &str) -> Result<Vec<(String, String)>, DecisionProjectionError> {
    let value: serde_json::Value = serde_json::from_str(text)
        .map_err(|_| DecisionProjectionError::Declaration(OBJECT_OF_STRINGS))?;
    let Some(object) = value.as_object() else {
        return Err(DecisionProjectionError::Declaration(OBJECT_OF_STRINGS));
    };
    object
        .iter()
        .map(|(key, item)| {
            item.as_str()
                .map(|text| (key.clone(), text.to_owned()))
                .ok_or(DecisionProjectionError::Declaration(OBJECT_OF_STRINGS))
        })
        .collect()
}

fn refused(error: DecisionProjectionError) -> PyResult<(Option<String>, Option<String>)> {
    match error {
        DecisionProjectionError::Declaration(message) => Err(crate::value_err(message)),
        DecisionProjectionError::Compiler(error) => Err(crate::py_err(error)),
        DecisionProjectionError::Refusal(refusal) => {
            Ok((None, Some(external_refusal_json(&refusal))))
        }
    }
}

fn projection_json(projected: &MultiArmMeanProjection) -> String {
    let coordinates =
        projected.coordinates.iter().map(ScientificQuantityWire::from).collect::<Vec<_>>();
    serde_json::json!({
        "outcome": projected.outcome,
        "design_id": projected.design_id,
        "snapshot_id": projected.snapshot_id,
        "program_id": projected.program_id,
        "labels": projected.labels,
        "coordinates": coordinates,
        "means": projected.means,
        "provider_id": MULTI_ARM_PROVIDER_ID,
        "rng_id": MULTI_ARM_RNG_ID,
    })
    .to_string()
}
