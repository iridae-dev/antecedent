//! Original scalar effect coordinates for the public result-to-decision handoff.
//!
//! A contrast is one scientific quantity, never two reconstructed outcome means.

use pyo3::prelude::*;

#[derive(Clone, Debug)]
pub(crate) struct NativeEffectAuthority {
    pub(crate) query: antecedent_core::AverageEffectQuery,
    pub(crate) status: antecedent_core::IdentificationStatus,
    pub(crate) ate: f64,
    pub(crate) has_structural_response: bool,
    pub(crate) support_status: Option<antecedent::support::CellStatus>,
    pub(crate) executed_contract: Option<antecedent::ExecutedContract>,
    pub(crate) names: Vec<String>,
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
            if units.trim().is_empty() || units.len() > 256 || population != "target" {
                return Err(crate::value_err(
                    "effect units must be declared (at most 256 bytes) and population must be target (the original observed sample)",
                ));
            }
            let refuse = |detail: &str| {
                (None, Some(serde_json::json!({
                "code": if detail == antecedent_core::reason_details::native_claims::UNSUPPORTED_ESTIMAND { "route_not_supported" } else { "invalid_argument" }, "stage": "bind", "detail": detail,
                "offending": null, "expected": "unchanged original static mean ATE",
                "supplied": "scalar analysis", "remedy": "use the original checked ATE contrast"
            }).to_string()))
            };
            let Some(authority) = &self.effect_authority else {
                return Ok(refuse(
                    antecedent_core::reason_details::native_claims::UNSUPPORTED_ESTIMAND,
                ));
            };
            let query = &authority.query;
            if authority.status
                != antecedent_core::IdentificationStatus::NonparametricallyIdentified
                || query.target_population != antecedent_core::TargetPopulation::AllObserved
                || query.outcome_functional != antecedent_core::OutcomeFunctional::Mean
                || !query.effect_modifiers.is_empty()
                || authority.has_structural_response
                || authority
                    .support_status
                    .is_some_and(|s| s != antecedent::support::CellStatus::Supported)
            {
                return Ok(refuse(
                    antecedent_core::reason_details::native_claims::UNSUPPORTED_ESTIMAND,
                ));
            }
            if !authority.ate.is_finite() || supplied_value.to_bits() != authority.ate.to_bits() {
                return Ok(refuse(
                    antecedent_core::reason_details::native_claims::PROJECTION_MISMATCH,
                ));
            }
            let Some(executed) = &authority.executed_contract else {
                return Ok(refuse(
                    antecedent_core::reason_details::native_claims::NATIVE_STATE_UNAVAILABLE,
                ));
            };
            let level = |intervention: &antecedent_core::Intervention| match intervention {
                antecedent_core::Intervention::Set { variable, value }
                    if *variable == query.treatment =>
                {
                    value.as_f64()
                }
                _ => None,
            };
            let (Some(control), Some(active)) = (level(&query.control), level(&query.active))
            else {
                return Ok(refuse(
                    antecedent_core::reason_details::native_claims::UNSUPPORTED_ESTIMAND,
                ));
            };
            let treatment = authority
                .names
                .get(query.treatment.raw() as usize)
                .ok_or_else(|| crate::value_err("original treatment identity unavailable"))?;
            let outcome = authority
                .names
                .get(query.outcome.raw() as usize)
                .ok_or_else(|| crate::value_err("original outcome identity unavailable"))?;
            Ok((Some(serde_json::json!({
            "coordinates": [{
                "version": 1, "variable_id": outcome, "variable_name": outcome,
                "role": "outcome", "units": units, "population_id": population,
                "regime_id": format!("contrast:do({treatment}={active})-do({treatment}={control})"),
                "horizon": 0, "functional_id": "mean_difference", "conditioning": [],
                "transform_id": "identity"
            }],
            "means": [authority.ate], "provider_id": "native:static_ate",
            "snapshot_id": executed.identities.data_snapshot.to_hex(),
            "causal_contract_id": executed.identities.program.unwrap_or(executed.identities.identification).to_hex(),
            "query": {
                "treatment": treatment, "outcome": outcome, "control": control, "active": active,
                "adjustment": self.adjustment_set
            },
            "rng_id": "none:mean_grid"
        }).to_string()), None))
        })
    }
}
