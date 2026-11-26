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
        Err(e) if e.is_cancelled() => Err(crate::CausalCancelledError::new_err(e.to_string())),
        Err(e) if e.is_resource_refusal() => {
            Err(crate::CausalResourceError::new_err(e.to_string()))
        }
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
            _ => None
        }
    }

    fn confidence_interval_json(&self) -> PyResult<Option<String>> {
        match &self.inner {
            BridgeImageResult::TrialConfidence(v) => {
                let current = v.is_calibrated();
                Ok(Some(json!({
                    "lower":v.interval.endpoints[0], "upper":v.interval.endpoints[1],"estimate":v.interval.point,
                    "level":v.interval.basis.key.level, "design":v.interval.basis.key.dependence,
                    "reading":if current { "sampling_confidence" } else { "sampling_nominal" },
                    "calibration_status":v.calibration.status.as_str(),
                    "coverage_record":v.calibration.record_id.as_ref().filter(|_| current),
                    "calibration_sha":v.calibration.calibration_sha.as_ref().filter(|_| current),
                }).to_string()))
            }
            _ => Ok(None),
        }
    }
    fn interval_quantity_json(&self) -> Option<String> {
        match &self.inner {
            BridgeImageResult::JointPosterior(v) => Some(quantity_json(&v.quantity).to_string()),
            BridgeImageResult::TrialConfidence(v) => {
                Some(quantity_json(&v.interval.quantity).to_string())
            }
            _ => None,
        }
    }
    fn threshold_decision_json(&self) -> Option<String> {
        if let BridgeImageResult::IntervalThreshold(v) = &self.inner {
            let mut fields = json!(v);
            fields["interval_identity"] =
                json!(SemanticDigest::from_bytes(v.interval_identity).to_hex());
            fields["identity"] = json!(self.identity());
            fields["decision_calibration"] = json!(v.decision_calibration());
            return Some(fields.to_string());
        }
        None
    }
    fn credible_mass(&self) -> Option<(f64, f64, f64, f64)> {
        match &self.inner {
            BridgeImageResult::JointPosterior(v) => {
                Some((v.endpoints[0], v.endpoints[1], v.level, v.attained_mass))
            }
            _ => None,
        }
    }
    fn distribution(&self) -> Option<PyJointDistributionArtifact> {
        match &self.inner {
            BridgeImageResult::JointPosterior(v) => Some(PyJointDistributionArtifact {
                artifact: v.posterior.clone(),
                source_claim: None,
            }),
            BridgeImageResult::TrialConfidence(v) => Some(PyJointDistributionArtifact {
                artifact: v.interval.bootstrap.clone(),
                source_claim: None,
            }),
            BridgeImageResult::IdentifiedSet(_) | BridgeImageResult::IntervalThreshold(_) => None,
            BridgeImageResult::JointOutcomes(v) => Some(PyJointDistributionArtifact {
                artifact: v.distribution.clone(),
                source_claim: None,
            }),
        }
    }
}

#[pymethods]
impl ImageProgram {
    #[pyo3(signature = (quantity,threshold,*,seed=1,threads=None,cancel=None,memory_bytes=None))]
    fn compile_threshold(
        &self,
        py: Python<'_>,
        quantity: &str,
        threshold: f64,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<crate::PyCancellationToken>,
        memory_bytes: Option<u64>,
    ) -> Outcome<ImageProgram> {
        let quantity = parse_quantity(&parse(quantity, "threshold coordinate")?)?;
        let ctx = crate::chain_decision_api::context(seed, threads, cancel, memory_bytes);
        crate::detach_catch(py, || {
            outcome(
                self.inner
                    .compile_threshold(quantity, threshold, &ctx)
                    .map(|inner| ImageProgram { inner }),
            )
        })
    }
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
#[pyo3(signature = (contract_json, study, replicates, level, bootstrap_seed, release_design=None, *, seed=1, threads=None, cancel=None, memory_bytes=None))]
fn compile_trial_bridge_confidence_interval(
    py: Python<'_>,
    contract_json: &str,
    study: Option<(String, Vec<f64>)>,
    replicates: u32,
    level: f64,
    bootstrap_seed: u64,
    release_design: Option<&str>,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
    memory_bytes: Option<u64>,
) -> Outcome<ImageProgram> {
    use antecedent::analysis::kernel_programs::bridge_images::{
        TrialBridgeExperiment, TrialBridgeReleaseDesign, TrialMeanStudy,
    };
    let study = match study {
        Some((study_id, surrogate_effects)) => {
            TrialMeanStudy::Independent { study_id, surrogate_effects }
        }
        None => TrialMeanStudy::Shared,
    };
    let release_design = release_design
        .map(|design| match design {
            "shared_uniform_gaussian_v1" => Ok(TrialBridgeReleaseDesign::SharedUniformGaussianV1),
            "independent_shifted_uniform_v1" => {
                Ok(TrialBridgeReleaseDesign::IndependentShiftedUniformV1)
            }
            _ => Err(crate::value_err("unknown trial bridge release design")),
        })
        .transpose()?;
    let request = BridgeImageRequest::TrialConfidence(Box::new(TrialBridgeExperiment {
        study,
        replicates,
        level,
        seed: bootstrap_seed,
        release_design,
        contract: parse_contract(&parse(contract_json, "trial bridge contract")?)?,
    }));
    let ctx = crate::chain_decision_api::context(seed, threads, cancel, memory_bytes);
    crate::detach_catch(py, || {
        outcome(
            antecedent::Study::compile_bridge_image(request, &ctx)
                .map(|inner| ImageProgram { inner }),
        )
    })
}

#[pyfunction]
#[pyo3(signature = (distribution, contract_json, intercept_shift, slope_shift, mapping_domain, level, *, seed=1, threads=None, cancel=None, memory_bytes=None))]
fn compile_posterior_bridge_interval(
    py: Python<'_>,
    distribution: PyRef<'_, PyJointDistributionArtifact>,
    contract_json: &str,
    intercept_shift: &str,
    slope_shift: &str,
    mapping_domain: (f64, f64),
    level: f64,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
    memory_bytes: Option<u64>,
) -> Outcome<ImageProgram> {
    let map = antecedent::analysis::kernel_programs::bridge_images::PosteriorAffineBridge {
        contract: parse_contract(&parse(contract_json, "bridge contract")?)?,
        intercept_shift: parse_quantity(&parse(intercept_shift, "intercept shift")?)?,
        slope_shift: parse_quantity(&parse(slope_shift, "slope shift")?)?,
        domain: domain(mapping_domain),
    };
    let request = BridgeImageRequest::JointPosterior {
        map: Box::new(map),
        source: distribution.inner().clone(),
        level,
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
    module.add_function(wrap_pyfunction!(compile_trial_bridge_confidence_interval, module)?)?;
    module.add_function(wrap_pyfunction!(compile_posterior_bridge_interval, module)?)?;
    module.add_class::<ImageResult>()?;
    module.add_function(wrap_pyfunction!(compile_identified_set_image, module)?)?;
    module.add_function(wrap_pyfunction!(compile_joint_outcome_image, module)?)?;
    module.add_function(wrap_pyfunction!(consume_scientific_bridge_image, module)?)?;
    Ok(())
}
