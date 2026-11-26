//! The compiled transport of a retained fitted effect to a cohort of another population.
//!
//! [`transport_cohort_effect_compiled`] takes what `transport_cohort_effect` takes, binds the
//! retained result, its population, the cohort and the selection-shift declaration into the
//! `TransportedCohortEffect` program family (`Study::compile_transported_cohort`) and executes
//! it: the program verifies the result, checks every premise in the route's order and issues the
//! transported cohort mean through the same route the direct entry calls. The summary, the
//! decision source for the same cohort, its artifact and every refusal are the direct entry's;
//! the summary adds a `program` section with the plan route and the lowered program identity.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::Study;
use antecedent::analysis::compiled_transported_cohort::TransportedCohortRequest;
use pyo3::prelude::*;

use super::{
    TransportBuildPayload, parse_cohort, refusal_json, selection_shift_of, summary,
    transport_refusal_json, transport_summary,
};
use crate::detach_catch;

/// Carry the retained model of an exported analysis result to a cohort of another population
/// through the compiled program, and issue the decision source for the same cohort:
/// `(transport summary json, decision source summary json, decision source artifact bytes,
/// refusal json)`.
#[pyfunction]
#[pyo3(signature = (model_bytes, expected_claim, source_population, cohort_json, outcome_units, shift_json, artifact_id, *, threads=None, cancel=None, memory_bytes=None))]
#[allow(
    clippy::too_many_arguments,
    reason = "the arguments of the direct entry, which this entry replaces"
)]
pub(super) fn transport_cohort_effect_compiled(
    py: Python<'_>,
    model_bytes: &[u8],
    expected_claim: Option<String>,
    source_population: &str,
    cohort_json: &str,
    outcome_units: &str,
    shift_json: Option<&str>,
    artifact_id: &str,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
    memory_bytes: Option<u64>,
) -> PyResult<TransportBuildPayload> {
    let cohort = match parse_cohort(cohort_json)? {
        Ok(cohort) => cohort,
        Err(error) => return Ok((None, None, None, Some(refusal_json(&error, "bind")))),
    };
    let shift = selection_shift_of(shift_json)?;
    let mut request =
        TransportedCohortRequest::new(model_bytes.to_vec(), source_population, cohort, shift);
    if let Some(claim) = expected_claim {
        request = request.expecting_claim(claim);
    }
    let units = outcome_units.to_owned();
    let artifact_id = artifact_id.to_owned();
    // Seed 0 is the seed this transport has always used. Threads, cancellation and memory
    // are the caller's; they do not enter the transported value.
    let ctx = crate::chain_decision_api::context(0, threads, cancel, memory_bytes);
    detach_catch(py, move || {
        let compiled = match Study::compile_transported_cohort(request) {
            Ok(compiled) => compiled,
            Err(error) => {
                return Ok((None, None, None, Some(transport_refusal_json(&error, "transport"))));
            }
        };
        let transported = match compiled.execute(&ctx) {
            Ok(transported) => transported,
            Err(error) => {
                return Ok((None, None, None, Some(transport_refusal_json(&error, "transport"))));
            }
        };
        let source = match transported.decision_source(&units, &ctx) {
            Ok(source) => source,
            Err(error) => {
                return Ok((None, None, None, Some(transport_refusal_json(&error, "bind"))));
            }
        };
        match source.to_bytes(&artifact_id) {
            Ok(bytes) => {
                let mut wire = transport_summary(&transported);
                wire["program"] = serde_json::json!({
                    "route": compiled.plan_view().route().as_str(),
                    "lowered_program_identity": compiled.lowered().identity().to_hex(),
                });
                Ok((Some(wire.to_string()), Some(summary(&source).to_string()), Some(bytes), None))
            }
            Err(error) => Ok((None, None, None, Some(refusal_json(&error, "bind")))),
        }
    })
}
