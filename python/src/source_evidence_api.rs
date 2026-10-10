//! Original native source evidence; independent imports grant no execution authority.
use antecedent::analysis::source_evidence::{SourceEvidence, SourceEvidenceError};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn invalid(detail: &str, message: impl AsRef<str>) -> PyErr {
    crate::with_reason_code(
        crate::value_err(format!("{detail}: {}", message.as_ref())),
        antecedent_core::reason_code!("invalid_argument"),
    )
}
fn error(cause: SourceEvidenceError) -> PyErr {
    invalid(cause.detail, cause.to_string())
}
#[pyclass(name = "SourceEvidenceHandle", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct PySourceEvidence {
    pub(crate) inner: SourceEvidence,
    pub(crate) resolution: Option<serde_json::Value>,
}
impl PySourceEvidence {
    pub(crate) fn from_claim(
        claim: &crate::program_claims_api::PyNativeResponseClaim,
    ) -> PyResult<Self> {
        Self::from_claim_named(claim, "native-source-evidence")
    }
    pub(crate) fn from_external(
        claim: &crate::external_api::PyExternalClaimArtifact,
    ) -> PyResult<Self> {
        let bytes = claim.artifact.to_bytes("external-source-evidence").map_err(|cause| {
            invalid("source_evidence.source_encoding_failed", cause.to_string())
        })?;
        SourceEvidence::from_external(&bytes)
            .map(|inner| Self { inner, resolution: None })
            .map_err(error)
    }
    fn from_claim_named(
        claim: &crate::program_claims_api::PyNativeResponseClaim,
        artifact_id: &str,
    ) -> PyResult<Self> {
        let coordinates = claim.claim.coordinates();
        let first = coordinates.first().ok_or_else(|| {
            invalid(
                "source_evidence.coordinates_unavailable",
                "original response has no semantic coordinates",
            )
        })?;
        let labels = antecedent_core::ResponseCoordinateLabels {
            outcome_units: &first.units,
            population_id: &first.population_id,
            transform_id: &first.transform_id,
        };
        let bytes = claim
            .authority
            .prepared
            .encode_contracted_result_with_quantity_labels(
                &claim.authority.result,
                artifact_id,
                &claim.authority.context,
                Some(&labels),
            )
            .map_err(|cause| {
                invalid("source_evidence.source_encoding_failed", cause.to_string())
            })?;
        SourceEvidence::from_result(&bytes)
            .map(|inner| Self { inner, resolution: None })
            .map_err(error)
    }
}
#[pymethods]
impl PySourceEvidence {
    #[getter]
    fn summary_json(&self) -> String {
        let mut summary = self.inner.summary();
        summary["resolution"] = self.resolution.clone().unwrap_or(serde_json::Value::Null);
        summary.to_string()
    }
    fn resolve_with(
        &self,
        actual: &crate::program_claims_api::PyNativeResponseClaim,
    ) -> PyResult<Self> {
        if !self.inner.is_native_source() {
            return Err(invalid(
                "source_evidence.resolution_scope_unsupported",
                "external evidence cannot be resolved by native authority",
            ));
        }
        if self
            .inner
            .unresolved()
            .iter()
            .any(|dependency| dependency.as_ref() != "dependencies.checked_response_grid_operation")
        {
            return Err(invalid(
                "source_evidence.resolution_scope_unsupported",
                "original unresolved dependency requires its own native operation resolver",
            ));
        }
        let fresh = Self::from_claim_named(actual, self.inner.artifact_id())?;
        self.inner.require_same_source(&fresh.inner).map_err(error)?;
        let executed =
            actual.authority.result.executed_contract.as_ref().expect("issued executed contract");
        let execution = antecedent_io::execution_digest(
            &antecedent_io::execution_identity_from_context(&actual.authority.context),
        )
        .map_err(|cause| invalid("source_evidence.source_encoding_failed", cause.to_string()))?;
        Ok(Self {
            inner: self.inner.clone(),
            resolution: Some(
                serde_json::json!({"kind":"original_native_response_reexecution","execution_id":execution.to_hex(),"data_snapshot_id":executed.identities.data_snapshot.to_hex(),"program_id":executed.identities.program.map(|id|id.to_hex()),"resolved_dependencies":self.inner.unresolved().iter().map(AsRef::as_ref).collect::<Vec<_>>(),"native_authority_issued":false,"calibration_license_issued":false}),
            ),
        })
    }
    fn diagnostics_at(&self, quantity_json: &str) -> PyResult<String> {
        if quantity_json.len() > 65536 {
            return Err(invalid(
                "source_evidence.limits_exceeded",
                "coordinate declaration exceeds its bounded schema",
            ));
        }
        let wire: ScientificQuantityWire = serde_json::from_str(quantity_json).map_err(|_| {
            invalid(
                "source_evidence.invalid_coordinates",
                "invalid semantic coordinate declaration",
            )
        })?;
        let quantity = antecedent_core::ScientificQuantity::try_from(wire).map_err(|_| {
            invalid(
                "source_evidence.invalid_coordinates",
                "invalid semantic coordinate declaration",
            )
        })?;
        self.inner.diagnostics_at(&quantity).map(|value| value.to_string()).map_err(error)
    }
    fn project(&self, contract_json: &str, actions: Vec<String>) -> PyResult<Self> {
        self.inner
            .project(contract_json, &actions)
            .map(|inner| Self { inner, resolution: self.resolution.clone() })
            .map_err(error)
    }
    fn export<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.inner.export().map(|bytes| PyBytes::new(py, &bytes)).map_err(error)
    }
    fn original_artifact<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.inner.original_bytes())
    }
    #[staticmethod]
    fn consume(bytes: &[u8]) -> PyResult<Self> {
        SourceEvidence::consume(bytes).map(|inner| Self { inner, resolution: None }).map_err(error)
    }
}
/// Mean-only native forward payload with its actual retained original issuer.
#[pyclass(name = "SourceBackedMeanClaim", skip_from_py_object)]
pub(crate) struct PySourceBackedMeanClaim {
    pub(crate) means: antecedent_design::decision_eval::MeanSource,
    pub(crate) original: crate::program_claims_api::PyNativeResponseClaim,
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeMeanWire {
    coordinates: Vec<ScientificQuantityWire>,
    means: Vec<f64>,
    provider_id: String,
    snapshot_id: String,
    causal_contract_id: String,
    rng_id: String,
}
/// Preserve a named mean adapter's scope while retaining the original opaque source.
#[pyfunction]
fn source_backed_native_mean(
    original: &crate::program_claims_api::PyNativeResponseClaim,
    declaration: &str,
) -> PyResult<PySourceBackedMeanClaim> {
    use antecedent::analysis::native_claims::NativeDecisionSource;
    use antecedent_design::decision_contract::{SourceRepresentation, SourceRequirement};
    if declaration.len() > 1024 * 1024 {
        return Err(invalid("source_evidence.limits_exceeded", "mean declaration exceeds bound"));
    }
    let wire: NativeMeanWire = serde_json::from_str(declaration)
        .map_err(|cause| invalid("source_evidence.point_binding_mismatch", cause.to_string()))?;
    let checked = original
        .claim
        .decision_source(&SourceRequirement {
            any_of: vec![SourceRepresentation::Mean],
            sampled_needs_error_receipt: false,
        })
        .map_err(|cause| invalid("source_evidence.point_binding_mismatch", cause.detail))?;
    let NativeDecisionSource::Mean(actual) = checked.source else {
        return Err(invalid("source_evidence.point_binding_mismatch", "original mean unavailable"));
    };
    let coordinates = wire
        .coordinates
        .into_iter()
        .map(antecedent_core::ScientificQuantity::try_from)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|cause| invalid("source_evidence.point_binding_mismatch", cause))?;
    if coordinates.len() != wire.means.len()
        || wire.provider_id != actual.provider_id
        || wire.snapshot_id != actual.snapshot_id
        || wire.causal_contract_id != actual.causal_contract_id
        || wire.rng_id != actual.rng_id
        || coordinates.iter().zip(&wire.means).any(|(quantity, value)| {
            actual
                .coordinates
                .iter()
                .position(|q| q.require_same_coordinate(quantity).is_ok())
                .is_none_or(|index| actual.means[index].to_bits() != value.to_bits())
        })
    {
        return Err(invalid(
            "source_evidence.point_binding_mismatch",
            "mean declaration differs from actual original native source",
        ));
    }
    Ok(PySourceBackedMeanClaim {
        means: antecedent_design::decision_eval::MeanSource {
            coordinates,
            means: wire.means,
            provider_id: wire.provider_id,
            snapshot_id: wire.snapshot_id,
            causal_contract_id: wire.causal_contract_id,
            rng_id: wire.rng_id,
        },
        original: original.clone(),
    })
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySourceEvidence>()?;
    module.add_class::<PySourceBackedMeanClaim>()?;
    module.add_function(wrap_pyfunction!(source_backed_native_mean, module)?)
}
