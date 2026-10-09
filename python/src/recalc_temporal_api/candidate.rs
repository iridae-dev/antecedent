//! Internal checked-session handoff: replay the original source proof and both arms.
//! This envelope adds no numerical evaluator or licensed inference claim.
use super::{PyTemporalSession, error};
use crate::recalc_api::{invalid, parse_json};
use antecedent::analysis::recalc_temporal::consume_temporal_recalc_artifact;
use antecedent_core::{ExecutionContext, IdentityDomain};
use antecedent_estimate::temporal_dependent_interval::DependentIntervalConfig;
use antecedent_io::temporal_interval_artifact::{IntervalConfigWire, IntervalResultWire};
use antecedent_io::temporal_recalc_artifact::{TemporalFunctionalWire, TemporalRecalcArtifactWire};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::{Deserialize, Serialize};

const MAX_SOURCE_BYTES: usize = 16 * 1024 * 1024;
const MAX_ARTIFACT_BYTES: usize = 24 * 1024 * 1024;
const MAX_HISTORY_REFITS: usize = 50_000_000;
const FEATURE: &str = "checked_temporal_interval_candidate_v1";
type Outcome = PyResult<(Option<NativeCheckedTemporalIntervalCandidate>, Option<String>)>;

#[derive(Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ExpectedIdentity {
    seal: String,
    source_data_digest: String,
    source_premises_digest: String,
    snapshot_id: String,
    initial_state_id: String,
    functional: TemporalFunctionalWire,
    producing_seed: u64,
    config: IntervalConfigWire,
    panel_digest: String,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidateWire {
    version: u32,
    required_features: Vec<String>,
    calibration: String,
    claim: String,
    identity: ExpectedIdentity,
    source: Vec<u8>,
    config: IntervalConfigWire,
    result: IntervalResultWire,
}
impl CandidateWire {
    fn seal(&self) -> PyResult<String> {
        antecedent_io::identity::digest_wire(
            IdentityDomain::Claim,
            &(
                self.version,
                &self.required_features,
                &self.calibration,
                &self.claim,
                &self.source,
                &self.config,
                &self.result,
            ),
        )
        .map(|digest| digest.to_hex())
        .map_err(crate::py_msg)
    }
    fn export_bytes(&self) -> PyResult<Vec<u8>> {
        let bytes = antecedent_io::to_cbor(self).map_err(crate::py_msg)?;
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(invalid(
                "recalc.limits_exceeded",
                "checked temporal interval artifact exceeds 24 MiB",
            ));
        }
        Ok(bytes)
    }
}

#[pyclass]
pub(super) struct NativeCheckedTemporalIntervalCandidate {
    bytes: std::sync::Arc<[u8]>,
    payload: String,
}
impl NativeCheckedTemporalIntervalCandidate {
    fn from_wire(wire: CandidateWire, bytes: std::sync::Arc<[u8]>) -> PyResult<Self> {
        // Raw input/proof bytes belong to the opaque export, not public metadata.
        let payload = serde_json::to_string(&serde_json::json!({
            "version":wire.version,"required_features":wire.required_features,
            "calibration":wire.calibration,"claim":wire.claim,
            "identity":wire.identity,"config":wire.config,"result":wire.result,
        }))
        .map_err(crate::py_msg)?;
        Ok(Self { bytes, payload })
    }
}
#[pymethods]
impl NativeCheckedTemporalIntervalCandidate {
    fn payload(&self) -> &str {
        &self.payload
    }
    fn export<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.bytes)
    }
}

fn before_copy(
    bytes: usize,
    memory: Option<u64>,
    cancel: Option<&crate::PyCancellationToken>,
) -> PyResult<()> {
    if cancel.is_some_and(|token| token.inner.is_cancelled()) {
        return Err(crate::with_reason_code(
            crate::CausalCancelledError::new_err(
                "recalc.cancelled: checked temporal interval cancelled before decode",
            ),
            antecedent_core::reason_code!("cancelled_no_claim"),
        ));
    }
    if memory
        .is_some_and(|limit| limit < (bytes as u64).saturating_mul(8).saturating_add(1024 * 1024))
    {
        return Err(invalid(
            "recalc.memory_budget_exceeded",
            "planned artifact decoding/copies exceed memory limit",
        ));
    }
    Ok(())
}

fn context(
    seed: u64,
    memory: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> ExecutionContext {
    let mut ctx = crate::py_execution_context_cancel(seed, 1, cancel);
    ctx.memory =
        antecedent_core::MemoryBudget { soft_limit_bytes: memory, hard_limit_bytes: memory };
    ctx
}
fn bounds(
    source: &TemporalRecalcArtifactWire,
    source_bytes: usize,
    config: &DependentIntervalConfig,
    max_units: usize,
    max_histories: usize,
    max_replicates: usize,
    memory: Option<u64>,
    ctx: &ExecutionContext,
) -> PyResult<()> {
    if config.replicates < 20
        || !(config.level > 0.0 && config.level < 1.0)
        || !(0.0..1.0).contains(&config.max_failed_fraction)
    {
        return Err(invalid(
            "recalc.invalid_declaration",
            "invalid checked interval configuration",
        ));
    }
    if ctx.cancellation.is_cancelled() {
        return Err(crate::with_reason_code(
            crate::CausalCancelledError::new_err(
                "recalc.cancelled: checked temporal interval cancelled before work",
            ),
            antecedent_core::reason_code!("cancelled_no_claim"),
        ));
    }
    let rows = source
        .request
        .units
        .iter()
        .try_fold(0usize, |count, unit| count.checked_add(unit.histories.len()));
    let rows = rows.ok_or_else(|| invalid("recalc.limits_exceeded", "history count overflow"))?;
    if max_units > 4096
        || max_histories > 100_000
        || max_replicates > 2000
        || source_bytes > MAX_SOURCE_BYTES
        || source.request.units.len() > max_units
        || rows > max_histories
        || config.replicates > max_replicates
        || rows.checked_mul(config.replicates).is_none_or(|work| work > MAX_HISTORY_REFITS)
    {
        return Err(invalid(
            "recalc.limits_exceeded",
            "checked temporal interval source or refit work exceeds bounds",
        ));
    }
    // Conservative bridge planning guard, not a claim of allocator accounting.
    // Includes concurrent source/wire copies, checked factor fitting and full receipts.
    let planned_bytes = (source_bytes as u64)
        .saturating_mul(8)
        .saturating_add((rows as u64).saturating_mul(1024))
        .saturating_add((source.request.units.len() as u64).saturating_mul(256))
        .saturating_add((config.replicates as u64).saturating_mul(256))
        .saturating_add(1024 * 1024);
    if memory.is_some_and(|limit| planned_bytes > limit) {
        return Err(crate::with_reason_code(
            crate::value_err(
                "recalc.memory_budget_exceeded: planned checked temporal interval working set exceeds limit",
            ),
            antecedent_core::reason_code!("invalid_argument"),
        ));
    }
    Ok(())
}
fn expected(
    source: &TemporalRecalcArtifactWire,
    seal: String,
    config: &IntervalConfigWire,
    panel_digest: &str,
) -> ExpectedIdentity {
    ExpectedIdentity {
        seal,
        source_data_digest: source.data_digest.clone(),
        source_premises_digest: source.premises_digest.clone(),
        snapshot_id: source.request.snapshot_id.clone(),
        initial_state_id: source.request.initial_state_id.clone(),
        functional: source.request.functional.clone(),
        producing_seed: source.seed,
        config: config.clone(),
        panel_digest: panel_digest.into(),
    }
}

pub(super) fn produce(
    session: &PyTemporalSession,
    py: Python<'_>,
    config_json: &str,
    memory: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> Outcome {
    let config_wire: IntervalConfigWire =
        parse_json(config_json, "checked temporal interval config")?;
    let config = config_wire.to_config().map_err(crate::py_msg)?;
    before_copy(0, memory, cancel.as_ref())?;
    crate::detach_catch(py, || {
        let bytes = match session.inner.export_result() {
            Ok(bytes) => bytes,
            Err(cause) => return Ok((None, Some(error(&cause)))),
        };
        if bytes.len() > MAX_SOURCE_BYTES {
            return Err(invalid(
                "recalc.limits_exceeded",
                "checked temporal source exceeds 16 MiB",
            ));
        }
        before_copy(bytes.len(), memory, cancel.as_ref())?;
        let source = TemporalRecalcArtifactWire::decode(&bytes).map_err(crate::py_msg)?;
        let ctx = context(source.seed, memory, cancel);
        bounds(&source, bytes.len(), &config, 4096, 100_000, 2000, memory, &ctx)?;
        let interval = match session.inner.candidate_interval_internal(&config, &ctx) {
            Ok(interval) => interval,
            Err(cause) => return Ok((None, Some(error(&cause)))),
        };
        let mut wire = CandidateWire {
            version: 1,
            required_features: vec![FEATURE.into()],
            calibration: "unmeasured".into(),
            claim: "dependence_preserving_calibration_unmeasured".into(),
            identity: expected(
                &source,
                String::new(),
                &config_wire,
                &format!("{:016x}", interval.panel_digest),
            ),
            source: bytes,
            config: config_wire,
            result: IntervalResultWire::from_interval(&interval),
        };
        wire.identity.seal = wire.seal()?;
        let bytes = std::sync::Arc::from(wire.export_bytes()?);
        Ok((Some(NativeCheckedTemporalIntervalCandidate::from_wire(wire, bytes)?), None))
    })
}

#[pyfunction]
#[pyo3(signature=(artifact,expected_identity_json,*,max_units=4096,max_histories=100_000,max_replicates=2000,memory_limit_bytes=None,cancel=None))]
#[allow(clippy::too_many_arguments)]
fn consume_checked_temporal_interval_candidate(
    py: Python<'_>,
    artifact: &[u8],
    expected_identity_json: &str,
    max_units: usize,
    max_histories: usize,
    max_replicates: usize,
    memory_limit_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> Outcome {
    if artifact.len() > MAX_ARTIFACT_BYTES {
        return Err(invalid(
            "recalc.limits_exceeded",
            "checked temporal interval artifact exceeds 24 MiB",
        ));
    }
    before_copy(artifact.len(), memory_limit_bytes, cancel.as_ref())?;
    if max_units > 4096 || max_histories > 100_000 || max_replicates > 2000 {
        return Err(invalid(
            "recalc.limits_exceeded",
            "consumer limits exceed checked handoff caps",
        ));
    }
    let bytes = artifact.to_vec();
    let expected_identity: ExpectedIdentity =
        parse_json(expected_identity_json, "expected checked temporal interval identity")?;
    crate::detach_catch(py, move || {
        let wire: CandidateWire = antecedent_io::from_cbor(&bytes)
            .map_err(|cause| crate::CausalSerializationError::new_err(cause.to_string()))?;
        if wire.version != 1
            || wire.required_features != [FEATURE]
            || wire.calibration != "unmeasured"
            || wire.claim != "dependence_preserving_calibration_unmeasured"
            || wire.identity != expected_identity
            || wire.identity.seal != wire.seal()?
        {
            return Err(invalid(
                "recalc.temporal_artifact_invalid",
                "checked temporal interval identity or envelope differs",
            ));
        }
        if wire.source.len() > MAX_SOURCE_BYTES {
            return Err(invalid(
                "recalc.limits_exceeded",
                "checked temporal source exceeds 16 MiB",
            ));
        }
        let source = TemporalRecalcArtifactWire::decode(&wire.source).map_err(crate::py_msg)?;
        if wire.identity
            != expected(
                &source,
                wire.identity.seal.clone(),
                &wire.config,
                &wire.result.panel_digest,
            )
        {
            return Err(invalid(
                "recalc.temporal_artifact_invalid",
                "source identity differs from retained expected identity",
            ));
        }
        let config = wire.config.to_config().map_err(crate::py_msg)?;
        let ctx = context(source.seed, memory_limit_bytes, cancel);
        bounds(
            &source,
            wire.source.len(),
            &config,
            max_units,
            max_histories,
            max_replicates,
            memory_limit_bytes,
            &ctx,
        )?;
        let (session, _) = match consume_temporal_recalc_artifact(&wire.source, &ctx) {
            Ok(replay) => replay,
            Err(cause) => return Ok((None, Some(error(&cause)))),
        };
        let interval = match session.candidate_interval_internal(&config, &ctx) {
            Ok(interval) => interval,
            Err(cause) => return Ok((None, Some(error(&cause)))),
        };
        // Canonical CBOR compares all receipt fields, preserving IEEE-754 bits.
        if antecedent_io::to_cbor(&wire.result).map_err(crate::py_msg)?
            != antecedent_io::to_cbor(&IntervalResultWire::from_interval(&interval))
                .map_err(crate::py_msg)?
        {
            return Err(invalid(
                "recalc.temporal_artifact_invalid",
                "checked temporal interval full independent replay differs",
            ));
        }
        if wire.export_bytes()? != bytes {
            return Err(invalid(
                "recalc.temporal_artifact_invalid",
                "checked interval envelope is not its canonical encoding",
            ));
        }
        Ok((
            Some(NativeCheckedTemporalIntervalCandidate::from_wire(
                wire,
                std::sync::Arc::from(bytes),
            )?),
            None,
        ))
    })
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeCheckedTemporalIntervalCandidate>()?;
    module.add_function(wrap_pyfunction!(consume_checked_temporal_interval_candidate, module)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent::analysis::recalc_temporal::{TemporalSession, execute_temporal_with_receipt};
    use antecedent_io::temporal_recalc_artifact::{
        TemporalHistoryWire, TemporalRecalcRequestWire, TemporalUnitWire,
    };

    fn fixture() -> PyTemporalSession {
        let units = (0..24u64)
            .map(|unit_id| {
                let mut histories = Vec::new();
                for s0 in 0..2 {
                    for a1 in 0..2 {
                        for l2 in 0..2 {
                            for a2 in 0..2 {
                                let index = u64::try_from(histories.len()).unwrap();
                                histories.push(TemporalHistoryWire {
                                    time_id: 3 * index,
                                    s0,
                                    a1,
                                    l2,
                                    a2,
                                    y: f64::from(u32::from(
                                        (unit_id + 7 * index) % 11 < 3 + u64::from(s0 + a1 + a2),
                                    )),
                                });
                            }
                        }
                    }
                }
                TemporalUnitWire { unit_id, histories }
            })
            .collect();
        let request = TemporalRecalcRequestWire {
            edges: (0..5).flat_map(|a| ((a + 1)..5).map(move |b| (a, b))).collect(),
            bidirected: vec![],
            selection_targets: vec![0],
            horizon: 2,
            lag_alignment: [0, 1, 2, 2, 2],
            period: (0, 49),
            units,
            snapshot_id: "checked-native-reseal-oracle".into(),
            initial_state: [0.3, 0.7],
            initial_state_id: "fixed-target-law".into(),
            functional: TemporalFunctionalWire::Effect { active: [1, 1], control: [0, 0] },
            benefit_per_unit: 1.0,
            cost: 0.0,
        };
        let mut inner = TemporalSession::new();
        execute_temporal_with_receipt(&mut inner, &request, &ExecutionContext::for_tests(771))
            .unwrap();
        PyTemporalSession { inner }
    }
    fn candidate(py: Python<'_>) -> NativeCheckedTemporalIntervalCandidate {
        let config = IntervalConfigWire::from_config(&DependentIntervalConfig {
            replicates: 20,
            seed: 901,
            method: antecedent_estimate::temporal_dependent_interval::IntervalMethod::Studentized,
            ..DependentIntervalConfig::default()
        });
        produce(&fixture(), py, &serde_json::to_string(&config).unwrap(), None, None)
            .unwrap()
            .0
            .unwrap()
    }
    fn reject_resealed(py: Python<'_>, mut wire: CandidateWire) {
        wire.identity.seal = wire.seal().unwrap();
        let expected = serde_json::to_string(&wire.identity).unwrap();
        let outcome = consume_checked_temporal_interval_candidate(
            py,
            &wire.export_bytes().unwrap(),
            &expected,
            4096,
            100_000,
            2000,
            None,
            None,
        );
        assert!(matches!(outcome, Err(_) | Ok((None, Some(_)))));
    }
    #[test]
    fn checked_interval_consumer_rejects_resealed_receipt_and_original_proof_mutations() {
        Python::initialize();
        Python::attach(|py| {
            let native = candidate(py);
            let original: CandidateWire = antecedent_io::from_cbor(&native.bytes).unwrap();
            let mut changed = original.clone();
            changed.result.studentization.as_mut().unwrap().pivots[0] = Some(123.0);
            reject_resealed(py, changed);
            let mut changed = original.clone();
            let mut source = TemporalRecalcArtifactWire::decode(&changed.source).unwrap();
            source.reports[0].mean += 0.01;
            changed.source = antecedent_io::to_cbor(&source).unwrap();
            reject_resealed(py, changed);
            let mut changed = original.clone();
            changed.config.seed += 1;
            changed.identity.config = changed.config.clone();
            reject_resealed(py, changed);
        });
    }
    #[test]
    fn checked_interval_consumer_rejects_point_only_artifact_substitution() {
        Python::initialize();
        Python::attach(|py| {
            let native = candidate(py);
            let wire: CandidateWire = antecedent_io::from_cbor(&native.bytes).unwrap();
            let expected = serde_json::to_string(&wire.identity).unwrap();
            let outcome = consume_checked_temporal_interval_candidate(
                py,
                &wire.source,
                &expected,
                4096,
                100_000,
                2000,
                None,
                None,
            );
            assert!(matches!(outcome, Err(_) | Ok((None, Some(_)))));
        });
    }
}
