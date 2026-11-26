//! Bind the same compiled exact-image programs used by Rust and by portable replay.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use antecedent::analysis::kernel_programs::bridge_images::{
    BridgeImageError, BridgeImageRequest, BridgeImageResult, CompiledBridgeImage, ExactImageError,
    JointOutcomeBridgeError, JointOutcomeMap, JointOutcomeTerm, consume_bridge_image,
};
use antecedent::contract::bridge::{FixedBridgeDomain, IdentifiedSetSource, ValueInterval};
use antecedent::query::IdentifiedSet;

type Outcome<T> = PyResult<(Option<T>, Option<String>)>;

fn outcome<T>(value: Result<T, ExactImageError>) -> Outcome<T> {
    match value {
        Ok(result) => Ok((Some(result), None)),
        Err(ExactImageError::Image(BridgeImageError::Mapping(
            JointOutcomeBridgeError::Contract(r),
        ))) => Ok((None, Some(external_refusal_json(&r)))),
        Err(ExactImageError::Support(r)) => Ok((
            None,
            Some(
                crate::bridge::RefusalJson::new(
                    r.code(),
                    "scientific_bridge",
                    r.detail,
                    &r.to_string(),
                )
                .json(),
            ),
        )),
        Err(ExactImageError::Image(
            BridgeImageError::Artifact(antecedent::io::IoError::Admission(r))
            | BridgeImageError::Mapping(JointOutcomeBridgeError::Artifact(
                antecedent::io::IoError::Admission(r),
            )),
        )) => Err(crate::CausalResourceError::new_err(r.to_string())),
        Err(ExactImageError::Image(e)) => Err(crate::serialization_err(e.to_string())),
        Err(ExactImageError::Compile(e)) => Err(crate::value_err(e.to_string())),
    }
}

fn domain(bounds: (f64, f64)) -> FixedBridgeDomain {
    FixedBridgeDomain(ValueInterval { low: bounds.0, high: bounds.1 })
}

#[pyclass(name = "CompiledBridgeImageHandle", skip_from_py_object)]
struct ImageProgram {
    inner: CompiledBridgeImage,
}

#[pyclass(name = "BridgeImageResultHandle", skip_from_py_object)]
struct ImageResult {
    inner: BridgeImageResult,
}

#[pymethods]
impl ImageResult {
    fn identity(&self) -> String {
        self.inner.identity().to_hex()
    }

    fn identified_set_json(&self) -> Option<String> {
        match &self.inner {
            BridgeImageResult::IdentifiedSet(v) => Some(json!({
                "quantity": quantity_json(&v.quantity), "lower": v.set.lower, "upper": v.set.upper
            }).to_string()),
            BridgeImageResult::JointOutcomes(_) => None
        }
    }

    fn distribution(&self) -> Option<PyJointDistributionArtifact> {
        match &self.inner {
            BridgeImageResult::IdentifiedSet(_) => None,
            BridgeImageResult::JointOutcomes(v) => Some(PyJointDistributionArtifact {
                artifact: v.distribution.clone(),
                source_claim: None,
            }),
        }
    }
}

#[pymethods]
impl ImageProgram {
    fn plan_json(&self) -> String {
        self.inner.plan_view().to_json().to_string()
    }
    fn program_identity(&self) -> String {
        self.inner.lowered().identity().to_hex()
    }
    fn unit_support(&self) -> Option<String> {
        crate::errors::unit_support_json(&self.inner.plan_view())
    }

    #[pyo3(signature = (*, seed=1, threads=None, cancel=None, memory_bytes=None))]
    fn execute(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<crate::PyCancellationToken>,
        memory_bytes: Option<u64>,
    ) -> Outcome<ImageResult> {
        let ctx = crate::chain_decision_api::context(seed, threads, cancel, memory_bytes);
        crate::detach_catch(py, || {
            let value = self.inner.execute(&ctx);
            if ctx.cancellation.is_cancelled() {
                return Err(crate::CausalCancelledError::new_err(
                    "bridge image execution cancelled",
                ));
            }
            outcome(value.map(|inner| ImageResult { inner }))
        })
    }

    #[pyo3(signature = (artifact_id, *, seed=1, threads=None, cancel=None, memory_bytes=None))]
    fn export<'py>(
        &self,
        py: Python<'py>,
        artifact_id: &str,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<crate::PyCancellationToken>,
        memory_bytes: Option<u64>,
    ) -> Outcome<(Bound<'py, PyBytes>, String)> {
        let ctx = crate::chain_decision_api::context(seed, threads, cancel, memory_bytes);
        let (bytes, refusal) = crate::detach_catch(py, || {
            let value = self.inner.export_with_identity(artifact_id, &ctx);
            if ctx.cancellation.is_cancelled() {
                return Err(crate::CausalCancelledError::new_err("bridge image export cancelled"));
            }
            outcome(value)
        })?;
        Ok((bytes.map(|(data, result)| (PyBytes::new(py, &data), result.to_hex())), refusal))
    }
}

#[pyfunction]
#[pyo3(signature = (contract_json, quantity, bounds, source_identity, mapping_domain, *, seed=1, threads=None, cancel=None, memory_bytes=None))]
fn compile_identified_set_image(
    py: Python<'_>,
    contract_json: &str,
    quantity: &str,
    bounds: (f64, f64),
    source_identity: &str,
    mapping_domain: (f64, f64),
    seed: u64,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
    memory_bytes: Option<u64>,
) -> Outcome<ImageProgram> {
    let request = BridgeImageRequest::IdentifiedSet {
        bridge: Box::new(parse_contract(&parse(contract_json, "bridge contract")?)?),
        domain: domain(mapping_domain),
        source: IdentifiedSetSource {
            quantity: parse_quantity(&parse(quantity, "source quantity")?)?,
            set: IdentifiedSet::try_new(bounds.0, bounds.1)
                .map_err(|e| crate::value_err(e.to_string()))?,
            claim_identity: digest(source_identity, "source_identity")?,
        },
    };
    let ctx = crate::chain_decision_api::context(seed, threads, cancel, memory_bytes);
    crate::detach_catch(py, || {
        outcome(
            antecedent::Study::compile_bridge_image(request, &ctx)
                .map(|inner| ImageProgram { inner }),
        )
    })
}

#[pyfunction]
#[pyo3(signature = (distribution, outputs, *, seed=1, threads=None, cancel=None, memory_bytes=None))]
fn compile_joint_outcome_image(
    py: Python<'_>,
    distribution: PyRef<'_, PyJointDistributionArtifact>,
    outputs: Vec<Vec<(String, (f64, f64))>>,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
    memory_bytes: Option<u64>,
) -> Outcome<ImageProgram> {
    let outputs = outputs
        .into_iter()
        .map(|terms| {
            terms
                .into_iter()
                .map(|(contract, bounds)| {
                    Ok(JointOutcomeTerm {
                        contract: parse_contract(&parse(&contract, "bridge contract")?)?,
                        domain: domain(bounds),
                    })
                })
                .collect::<PyResult<Vec<_>>>()
        })
        .collect::<PyResult<Vec<_>>>()?;
    let request = BridgeImageRequest::JointOutcomes {
        map: JointOutcomeMap { outputs },
        source: distribution.inner().clone(),
    };
    let ctx = crate::chain_decision_api::context(seed, threads, cancel, memory_bytes);
    crate::detach_catch(py, || {
        outcome(
            antecedent::Study::compile_bridge_image(request, &ctx)
                .map(|inner| ImageProgram { inner }),
        )
    })
}

#[pyfunction]
#[pyo3(signature = (artifact, program, result, *, seed=1, threads=None, cancel=None, memory_bytes=None))]
fn consume_scientific_bridge_image(
    py: Python<'_>,
    artifact: &[u8],
    program: &str,
    result: &str,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
    memory_bytes: Option<u64>,
) -> Outcome<ImageResult> {
    let ctx = crate::chain_decision_api::context(seed, threads, cancel, memory_bytes);
    let program = digest(program, "expected_program")?;
    let result = digest(result, "expected_result")?;
    crate::detach_catch(py, || {
        let value = consume_bridge_image(artifact, program, result, &ctx);
        if ctx.cancellation.is_cancelled() {
            return Err(crate::CausalCancelledError::new_err("bridge image replay cancelled"));
        }
        outcome(value.map(|inner| ImageResult { inner }))
    })
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ImageProgram>()?;
    module.add_class::<ImageResult>()?;
    module.add_function(wrap_pyfunction!(compile_identified_set_image, module)?)?;
    module.add_function(wrap_pyfunction!(compile_joint_outcome_image, module)?)?;
    module.add_function(wrap_pyfunction!(consume_scientific_bridge_image, module)?)?;
    Ok(())
}
