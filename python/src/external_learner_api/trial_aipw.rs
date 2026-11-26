//! Python learners on the compiled trial-to-target AIPW transport.
//!
//! [`trial_aipw_learner_transport`] declares two registered Python providers as the source
//! outcome and source-membership learners and runs the compiled `TrialAipwLearnerTransport`
//! program. Each provider fit returns its model as a portable prediction map.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::compiled_trial_aipw_learners::TrialAipwLearnersRequest;
use antecedent::estimate::TrialAipwOptions;
use antecedent::Study;
use pyo3::prelude::*;

use super::{callback_context, smoothed_dose::portable_factories, PythonReceipt};

/// Run the compiled trial AIPW transport with two registered Python providers as its learners.
///
/// `graph`, `selections`, `source`, `target`, `treatment`, `outcome`, `data` and `options` are
/// what `prepare_learned_trial` takes. `outcome_learner` is a `regression` learner and
/// `membership` a `binary_probability` learner. The JSON holds the contrast (without per-row
/// scores), the route, the lowered program identity, the claim and the calibration status.
#[pyfunction]
#[pyo3(signature = (
    graph, selections, source, target, treatment, outcome, data, options, outcome_learner,
    membership, *, identity, seed=1, cancel=None, memory_limit_bytes=None
))]
#[allow(
    clippy::too_many_arguments,
    reason = "the trial query and its learners are independent inputs"
)]
pub(super) fn trial_aipw_learner_transport(
    py: Python<'_>,
    graph: &crate::graphs::Admg,
    selections: Vec<String>,
    source: String,
    target: String,
    treatment: String,
    outcome: String,
    data: &Bound<'_, PyAny>,
    options: &str,
    outcome_learner: Py<PyAny>,
    membership: Py<PyAny>,
    identity: String,
    seed: u64,
    cancel: Option<crate::PyCancellationToken>,
    memory_limit_bytes: Option<u64>,
) -> PyResult<(String, Vec<PythonReceipt>)> {
    let (learners, state) = portable_factories(py, outcome_learner, membership, identity)?;
    let (diagram, query) = crate::learned_trial_api::trial_request(
        graph, selections, source, target, treatment, outcome,
    )?;
    let input = crate::learned_trial_api::parse_input(data, &graph.names)?;
    let options: TrialAipwOptions =
        serde_json::from_str(options).map_err(|e| crate::value_err(e.to_string()))?;
    let ctx = callback_context(seed, cancel, memory_limit_bytes)?;
    let result = py.detach(|| {
        let compiled = Study::compile_trial_aipw_learners(
            TrialAipwLearnersRequest::new(diagram, query, input, options, learners),
            &ctx,
        )
        .map_err(crate::transport_common::error)?;
        let run = compiled.execute(&ctx).map_err(crate::transport_common::error)?;
        let mut estimate = serde_json::to_value(&run.estimate)
            .map_err(|e| crate::serialization_err(e.to_string()))?;
        if let Some(map) = estimate.as_object_mut() {
            map.remove("scores");
            map.remove("membership");
            map.remove("mu0");
            map.remove("mu1");
        }
        Ok(serde_json::json!({
            "estimate": estimate,
            "route": compiled.plan_view().route().as_str(),
            "program_identity": run.lowered_program_identity.to_hex(),
            "claim": run.claim,
            "calibration": run.calibration,
        })
        .to_string())
    });
    state.finish(result)
}
