//! JSON read-model of the intervals an execution reported.
//!
//! The bindings are derived by the facade from the result's own content
//! (`StudyResult::reported_interval_bindings`), so they describe what the
//! execution reported, never what was requested. Python reads them as typed views;
//! it does not re-derive the method, dependence rule or replicate counts.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::StudyResult;
use antecedent::result::IntervalBinding;

fn binding_json(binding: &IntervalBinding) -> serde_json::Value {
    serde_json::json!({
        "method": binding.method.as_str(),
        "se_kind": binding.se_kind.map(|kind| kind.as_str()),
        "level": binding.level.is_finite().then_some(binding.level),
        "dependence": binding.dependence,
        "replicates_ok": binding.replicates_ok,
        "replicates_failed": binding.replicates_failed,
        "posterior_draws": binding.posterior_draws,
    })
}

/// Every reported interval of a result, primary first, as a JSON array.
///
/// The inference family is the facade's own post-hoc rule: a result that carries a
/// posterior was Bayesian.
pub(crate) fn bindings_json(result: &StudyResult) -> Option<String> {
    let bindings = result.reported_interval_bindings(result.posterior.is_some());
    Some(serde_json::Value::Array(bindings.iter().map(binding_json).collect()).to_string())
}
