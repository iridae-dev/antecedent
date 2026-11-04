//! Native bridge of the compiled declared-shift MNAR candidate.
use crate::{bridge::RefusalJson, detach_catch};
use antecedent::reason::reason_code;
use pyo3::prelude::*;

fn refusal_json(code: &str, detail: &str, message: &str) -> String {
    RefusalJson::new(code, "mnar_fixed_shift", detail, message).json()
}

/// A compiler-retained declared-shift MNAR candidate. Its interval is released only by
/// the measured consumer under an attesting coverage record.
#[pyclass(name = "NativePreparedMnarFixedShift")]
pub(crate) struct NativePreparedMnarFixedShift {
    inner: antecedent::analysis::mnar_fixed_shift::PreparedMnarFixedShift,
}

#[pymethods]
impl NativePreparedMnarFixedShift {
    /// Request the measured scalar; absent or inapplicable coverage refuses.
    #[pyo3(signature = (memory_bytes=None, cancel=None))]
    fn measured_interval(
        &self,
        py: Python<'_>,
        memory_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<(Option<crate::measured_inference_api::NativeMeasuredInference>, Option<String>)>
    {
        let ctx = crate::transport_common::execution_context(0, memory_bytes, cancel);
        let reading = detach_catch(py, || Ok(self.inner.measured_interval(&ctx)))?;
        match reading {
            Ok(value) => Ok((
                Some(crate::measured_inference_api::NativeMeasuredInference::from_io(value)?),
                None,
            )),
            Err(antecedent::io::IoError::Refused { code, message }) => {
                let (detail, text) = message.split_once(": ").unwrap_or((
                    antecedent::reason::reason_details::mnar_fixed_shift::INTERVAL_NOT_SUPPORTED,
                    &message,
                ));
                Ok((None, Some(refusal_json(code, detail, text))))
            }
            Err(error) => Ok((
                None,
                Some(refusal_json(
                    reason_code!("invalid_argument"),
                    antecedent::reason::reason_details::mnar_fixed_shift::INVALID_SOURCE,
                    &error.to_string(),
                )),
            )),
        }
    }
}

/// Compile a declared-shift candidate from its source, treated-arm shift and untested design premise.
#[pyfunction]
#[allow(clippy::float_cmp, reason = "treatment is exactly binary by declaration")]
#[allow(clippy::too_many_arguments, reason = "the Python signature mirrors the declared source")]
#[pyo3(signature = (subject_ids, treatment, outcome, population, treated_shift, memory_bytes=None, cancel=None))]
fn mnar_fixed_shift_prepare(
    py: Python<'_>,
    subject_ids: Vec<u64>,
    treatment: Vec<f64>,
    outcome: Vec<Option<f64>>,
    population: String,
    treated_shift: f64,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<(Option<NativePreparedMnarFixedShift>, Option<String>)> {
    use antecedent::schema::mnar_fixed_shift::{
        FixedShiftSource, SamplingPremise, SourceError, SourceRow,
    };
    if subject_ids.len() != treatment.len() || treatment.len() != outcome.len() {
        return Ok((
            None,
            Some(refusal_json(
                reason_code!("invalid_argument"),
                antecedent::reason::reason_details::mnar_fixed_shift::INVALID_SOURCE,
                "subject, treatment, and outcome row counts must match",
            )),
        ));
    }
    let mut rows = Vec::with_capacity(treatment.len());
    for ((subject_id, assignment), outcome) in subject_ids.into_iter().zip(treatment).zip(outcome) {
        let treated = if assignment == 1.0 {
            true
        } else if assignment == 0.0 {
            false
        } else {
            return Ok((
                None,
                Some(refusal_json(
                    reason_code!("invalid_argument"),
                    antecedent::reason::reason_details::mnar_fixed_shift::INVALID_SOURCE,
                    "treatment must be binary",
                )),
            ));
        };
        rows.push(SourceRow { subject_id, treated, outcome });
    }
    let source = match FixedShiftSource::new(
        population,
        0,
        1,
        SamplingPremise::IndependentRandomizedArms,
        treated_shift,
        rows,
    ) {
        Ok(source) => source,
        Err(error) => {
            let detail = match error {
                SourceError::InvalidDeclaration => {
                    antecedent::reason::reason_details::mnar_fixed_shift::INVALID_DECLARATION
                }
                SourceError::DuplicateSubject => {
                    antecedent::reason::reason_details::mnar_fixed_shift::DUPLICATE_SUBJECT
                }
                SourceError::InvalidOutcome => {
                    antecedent::reason::reason_details::mnar_fixed_shift::INVALID_OUTCOME
                }
                SourceError::InsufficientObservations => {
                    antecedent::reason::reason_details::mnar_fixed_shift::INSUFFICIENT_OBSERVATIONS
                }
                SourceError::ConstantOutcome => {
                    antecedent::reason::reason_details::mnar_fixed_shift::CONSTANT_OUTCOME
                }
                SourceError::InvalidShift => {
                    antecedent::reason::reason_details::mnar_fixed_shift::INVALID_SHIFT
                }
            };
            return Ok((
                None,
                Some(refusal_json(reason_code!("invalid_argument"), detail, &error.to_string())),
            ));
        }
    };
    let ctx = crate::transport_common::execution_context(0, memory_bytes, cancel);
    let prepared =
        detach_catch(py, move || Ok(antecedent::Study::prepare_mnar_fixed_shift(source, &ctx)))?;
    Ok(match prepared {
        Ok(inner) => (Some(NativePreparedMnarFixedShift { inner }), None),
        Err(error) => {
            let message = error.to_string();
            (
                None,
                Some(match &error {
                    antecedent::CausalError::Cancelled { .. } => refusal_json(
                        reason_code!("execution_cancelled"),
                        antecedent::reason::reason_details::mnar_fixed_shift::PREPARATION_CANCELLED,
                        &message,
                    ),
                    antecedent::CausalError::Resource { .. } => refusal_json(
                        reason_code!("execution_resource_limit"),
                        antecedent::reason::reason_details::mnar_fixed_shift::PREPARATION_RESOURCE_LIMIT,
                        &message,
                    ),
                    _ => refusal_json(
                        error.reason_code().unwrap_or("invalid_argument"),
                        antecedent::reason::reason_details::mnar_fixed_shift::PREPARATION,
                        &message,
                    ),
                }),
            )
        }
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativePreparedMnarFixedShift>()?;
    module.add_function(wrap_pyfunction!(mnar_fixed_shift_prepare, module)?)?;
    Ok(())
}
