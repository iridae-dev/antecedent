//! Python bindings for the temporal extensions of the finite two-step sequence
//! (2.3A, X5): the uncertain initial state and the new-period refresh (both point
//! only and licensed), and measured dependent intervals.
//!
//! Python builds the declarations; every identity, check and refusal rule is Rust's.
//! A refusal comes back as structured JSON for the Python layer to raise as its own
//! exception type.
use antecedent::{IntervalEstimand, TemporalInitialState, TemporalWindow};
use antecedent_core::reason_code;
use antecedent_estimate::temporal_dependent_interval::{
    DependentIntervalConfig, IntervalMethod, SequenceHistory, UnitHistories,
};
use antecedent_estimate::temporal_initial_state::{
    InitialStateLaw, InitialStatePopulation, InitialStateSpec,
};
use antecedent_estimate::temporal_refresh::ObservationPeriod;
use antecedent_io::IoError;
use antecedent_io::temporal_initial_state_artifact::{
    TemporalInitialStateConsumeLimits, TemporalPremisesWire,
};
use antecedent_io::temporal_refresh_artifact::{ReceiptWire, WindowIdentityWire};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::BTreeSet;

use crate::CausalSerializationError;

const INITIAL_STATE_PREFIX: &[u8] = b"ANTECEDENT-TEMPORAL-INITIAL-STATE\x01";
const REFRESH_PREFIX: &[u8] = b"ANTECEDENT-TEMPORAL-REFRESH\x01";

/// `(time_id, s0, a1, l2, a2, y)`.
type HistoryTuple = (u64, u32, u32, u32, u32, f64);
/// `(unit_id, histories)`.
type UnitTuple = (u64, Vec<HistoryTuple>);
/// `(population, snapshot_id, [(state, mass)])`.
type LawTuple = (String, String, Vec<(u32, f64)>);
/// `(state_variable, time_order, source_regime, target_regime, graph_id, proof_id)`.
type PremisesTuple = (String, Vec<String>, String, String, String, String);
/// `(horizon, lag_alignment, intervention_history, selection_targets, start, end, graph, proof)`.
type WindowTuple =
    (usize, Vec<(String, u8)>, Vec<String>, Vec<String>, i64, i64, Option<String>, Option<String>);
/// A payload or a structured refusal.
type Outcome<T> = PyResult<(Option<T>, Option<String>)>;

fn refusal_json(code: &str, message: &str) -> String {
    let (detail, text) = message.split_once(": ").unwrap_or(("", message));
    serde_json::json!({
        "code": code,
        "stage": "temporal_extension",
        "detail": detail,
        "message": text,
        "remedy": serde_json::Value::Null,
    })
    .to_string()
}

/// A producer outcome: a refusal is structured, any other failure is invalid input.
fn produce<T>(result: Result<T, IoError>) -> Outcome<T> {
    match result {
        Ok(value) => Ok((Some(value), None)),
        Err(IoError::Refused { code, message }) => Ok((None, Some(refusal_json(code, &message)))),
        Err(other) => Err(crate::value_err(other.to_string())),
    }
}

/// A consumer outcome: a refusal is structured, any other failure means the bytes
/// do not describe what they claim.
fn consume<T>(result: Result<T, IoError>) -> Outcome<T> {
    match result {
        Ok(value) => Ok((Some(value), None)),
        Err(IoError::Refused { code, message }) => Ok((None, Some(refusal_json(code, &message)))),
        Err(other) => Err(CausalSerializationError::new_err(other.to_string())),
    }
}

fn units_of(units: Option<Vec<UnitTuple>>) -> Option<Vec<UnitHistories>> {
    units.map(|units| {
        units
            .into_iter()
            .map(|(unit_id, histories)| UnitHistories {
                unit_id,
                histories: histories
                    .into_iter()
                    .map(|(time_id, s0, a1, l2, a2, y)| SequenceHistory {
                        time_id,
                        s0,
                        a1,
                        l2,
                        a2,
                        y,
                    })
                    .collect(),
            })
            .collect()
    })
}

fn panel_of(
    snapshot_id: &str,
    units: Option<Vec<UnitTuple>>,
) -> Result<antecedent_estimate::temporal_dependent_interval::TemporalUnitPanel, IoError> {
    Ok(antecedent_estimate::temporal_dependent_interval::TemporalUnitPanel::new(
        snapshot_id,
        units_of(units),
    )?)
}

fn spec_of(law: Option<LawTuple>, point: Option<u32>) -> Result<InitialStateSpec, IoError> {
    match (law, point) {
        (Some((population, snapshot, states)), None) => {
            let population = match population.as_str() {
                "target" => InitialStatePopulation::Target,
                "source" => InitialStatePopulation::Source,
                _ => {
                    return Err(IoError::Refused {
                        code: reason_code!("invalid_argument"),
                        message: format!(
                            "initial_state.invalid_law: population must be target or source, \
                             not {population}"
                        ),
                    });
                }
            };
            Ok(InitialStateSpec::Law(InitialStateLaw::new(population, snapshot, states)?))
        }
        (None, Some(s0)) => Ok(InitialStateSpec::Point(s0)),
        _ => Err(IoError::Refused {
            code: reason_code!("invalid_argument"),
            message: "initial_state.invalid_law: supply exactly one of a law or a point state"
                .into(),
        }),
    }
}

fn premises_of(premises: PremisesTuple) -> TemporalPremisesWire {
    let (initial_state_variable, time_order, source_regime, target_regime, graph_id, proof_id) =
        premises;
    TemporalPremisesWire {
        initial_state_variable,
        time_order,
        source_regime,
        target_regime,
        graph_id,
        proof_id,
    }
}

fn window_of(window: WindowTuple) -> TemporalWindow {
    let (horizon, lag_alignment, intervention_history, targets, start, end, graph_id, proof_id) =
        window;
    TemporalWindow {
        horizon,
        lag_alignment,
        intervention_history,
        selection_targets: targets.into_iter().collect::<BTreeSet<_>>(),
        period: ObservationPeriod { start, end },
        graph_id,
        proof_id,
    }
}

fn to_value(value: &impl serde::Serialize) -> serde_json::Value {
    serde_json::to_value(value).unwrap_or(serde_json::Value::Null)
}

fn state_payload(state: &TemporalInitialState) -> String {
    let marginalized = state.marginalized();
    let refresh = state.receipt().map(|receipt| {
        serde_json::json!({
            "receipt": to_value(&ReceiptWire::from_receipt(receipt)),
            "previous_value": state.previous_value(),
            "interval_invalidated": state.interval_invalidated(),
        })
    });
    serde_json::json!({
        "label": marginalized.label(),
        "value": marginalized.value,
        "sequence": state.sequence(),
        "contributions": marginalized.contributions.iter().map(|c| serde_json::json!({
            "state": c.s0,
            "mass": c.mass,
            "response": c.response,
        })).collect::<Vec<_>>(),
        "state_snapshot_id": marginalized.state_snapshot_id,
        "state_law_digest": format!("{:016x}", marginalized.state_law_digest),
        "panel_snapshot_id": marginalized.panel_snapshot_id,
        "inference_claim": marginalized.inference_claim,
        "fixed": state.fixed().map(|fixed| serde_json::json!({
            "label": fixed.label(),
            "state": fixed.s0,
            "value": fixed.value,
            "panel_snapshot_id": fixed.panel_snapshot_id,
        })),
        "premises": to_value(state.premises()),
        "window": state.identity().map(|id| to_value(&WindowIdentityWire::from_identity(id))),
        "refresh": refresh,
    })
    .to_string()
}

fn frame(prefix: &[u8], raw: &[u8]) -> Vec<u8> {
    let mut framed = prefix.to_vec();
    framed.extend_from_slice(raw);
    framed
}

fn unframe<'a>(prefix: &[u8], bytes: &'a [u8], what: &str) -> PyResult<&'a [u8]> {
    bytes
        .strip_prefix(prefix)
        .ok_or_else(|| CausalSerializationError::new_err(format!("invalid {what} artifact format")))
}

/// A prepared target-marginal initial-state result and the window it is valid for.
#[pyclass(skip_from_py_object)]
struct NativeTemporalInitialState {
    inner: TemporalInitialState,
}

#[pymethods]
impl NativeTemporalInitialState {
    /// The result as JSON.
    fn payload(&self) -> String {
        state_payload(&self.inner)
    }

    /// The initial-state artifact, framed.
    fn export(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let raw =
            self.inner.export().map_err(|e| CausalSerializationError::new_err(e.to_string()))?;
        Ok(PyBytes::new(py, &frame(INITIAL_STATE_PREFIX, &raw)).unbind())
    }

    /// The artifact of the refresh that produced this result, framed.
    fn export_refresh(&self, py: Python<'_>) -> Outcome<Py<PyBytes>> {
        produce(self.inner.export_refresh()).map(|(bytes, refusal)| {
            (bytes.map(|raw| PyBytes::new(py, &frame(REFRESH_PREFIX, &raw)).unbind()), refusal)
        })
    }

    /// Refresh onto a replacement panel of a new period.
    fn refresh(
        &self,
        snapshot_id: &str,
        units: Option<Vec<UnitTuple>>,
        window: WindowTuple,
        interval_existed: bool,
    ) -> Outcome<Self> {
        let result = panel_of(snapshot_id, units)
            .and_then(|panel| self.inner.refresh(panel, &window_of(window), interval_existed));
        produce(result.map(|inner| Self { inner }))
    }
}

/// Evaluate the target-marginal sum on a panel of repeated units.
#[pyfunction]
#[pyo3(signature = (snapshot_id, units, sequence, law, point, fixed_state, premises, window))]
#[allow(clippy::too_many_arguments)]
fn temporal_initial_state_prepare(
    snapshot_id: &str,
    units: Option<Vec<UnitTuple>>,
    sequence: (u32, u32),
    law: Option<LawTuple>,
    point: Option<u32>,
    fixed_state: Option<u32>,
    premises: PremisesTuple,
    window: Option<WindowTuple>,
) -> Outcome<NativeTemporalInitialState> {
    let result = panel_of(snapshot_id, units).and_then(|panel| {
        let spec = spec_of(law, point)?;
        let window = window.map(window_of);
        TemporalInitialState::prepare(
            premises_of(premises),
            [sequence.0, sequence.1],
            spec,
            fixed_state,
            panel,
            window.as_ref(),
        )
    });
    produce(result.map(|inner| NativeTemporalInitialState { inner }))
}

/// Independently replay a framed initial-state artifact.
#[pyfunction]
#[pyo3(signature = (artifact, *, max_summary_rows=1_000_000, max_units=100_000))]
fn consume_temporal_initial_state_artifact(
    artifact: &[u8],
    max_summary_rows: usize,
    max_units: usize,
) -> Outcome<String> {
    let raw = unframe(INITIAL_STATE_PREFIX, artifact, "temporal initial-state")?;
    let limits = TemporalInitialStateConsumeLimits { max_summary_rows, max_units };
    consume(antecedent::consume_temporal_initial_state_artifact(raw, &limits).map(
        |(wire, replay)| {
            serde_json::json!({
                "marginalized": to_value(&wire.marginalized),
                "fixed": to_value(&wire.fixed),
                "replayed_value": replay.value,
                "replayed_fixed": replay.fixed.map(|(state, value)| serde_json::json!({
                    "state": state,
                    "value": value,
                })),
                "law": to_value(&wire.law),
                "panel_snapshot_id": wire.panel.snapshot_id,
                "premises": to_value(&wire.premises),
                "sequence": wire.sequence,
                "inference_claim": wire.marginalized.inference_claim,
            })
            .to_string()
        },
    ))
}

/// Independently re-decide and replay a framed refresh artifact.
#[pyfunction]
#[pyo3(signature = (artifact, *, max_summary_rows=1_000_000, max_units=100_000))]
fn consume_temporal_refresh_artifact(
    artifact: &[u8],
    max_summary_rows: usize,
    max_units: usize,
) -> Outcome<String> {
    let raw = unframe(REFRESH_PREFIX, artifact, "temporal refresh")?;
    let limits = TemporalInitialStateConsumeLimits { max_summary_rows, max_units };
    consume(antecedent::consume_temporal_refresh_artifact(raw, &limits).map(|(wire, replay)| {
        serde_json::json!({
            "decision": wire.decision,
            "invalidation": wire.invalidation,
            "old_value": replay.old_value,
            "new_value": replay.new_value,
            "interval_invalidated": wire.interval_invalidated,
            "interval_present": wire.refreshed.as_ref().map(|r| r.interval_present),
            "label": wire.refreshed.as_ref().map(|r| r.label.clone()),
            "receipt": to_value(&wire.receipt),
            "old_window": to_value(&wire.old_window),
            "new_window": to_value(&wire.new_window),
            "law": to_value(&wire.law),
            "sequence": wire.sequence,
            "premises": to_value(&wire.premises),
            "inference_claim": "point_only",
        })
        .to_string()
    }))
}

/// The closed dependent-interval route: validates the arguments through the Rust
/// core, then returns the structured refusal. Never returns an interval.
#[pyfunction]
#[pyo3(signature = (snapshot_id, units, sequence, estimand, fixed_state, law, point, replicates, seed, level, method, min_units, max_failed_fraction))]
#[allow(clippy::too_many_arguments)]
fn temporal_dependent_interval_closed(
    snapshot_id: &str,
    units: Option<Vec<UnitTuple>>,
    sequence: (u32, u32),
    estimand: &str,
    fixed_state: Option<u32>,
    law: Option<LawTuple>,
    point: Option<u32>,
    replicates: usize,
    seed: u64,
    level: f64,
    method: &str,
    min_units: usize,
    max_failed_fraction: f64,
) -> PyResult<String> {
    let method = match method {
        "percentile" => IntervalMethod::Percentile,
        "basic" => IntervalMethod::Basic,
        "studentized" => IntervalMethod::Studentized,
        other => {
            return Err(crate::with_reason_code(
                crate::value_err(format!(
                    "interval method must be percentile, basic or studentized, not {other}"
                )),
                reason_code!("invalid_argument"),
            ));
        }
    };
    let config =
        DependentIntervalConfig { replicates, seed, level, method, min_units, max_failed_fraction };
    let estimand = match (estimand, fixed_state) {
        ("observed_initial_state", None) => Ok(IntervalEstimand::Observed),
        ("fixed_initial_state", Some(s0)) => Ok(IntervalEstimand::FixedState(s0)),
        ("marginalized_initial_state", None) => {
            spec_of(law, point).map(IntervalEstimand::Marginalized)
        }
        _ => Err(IoError::Refused {
            code: reason_code!("invalid_argument"),
            message: "temporal_interval.invalid_estimand: name observed, fixed (with a state) or \
                      marginalized (with a law)"
                .into(),
        }),
    };
    let sequence = [sequence.0, sequence.1];
    let outcome = estimand.and_then(|estimand| {
        antecedent::analysis::validate_temporal_dependent_interval(
            snapshot_id,
            units_of(units),
            sequence,
            estimand,
            &config,
        )
    });
    match outcome {
        Ok(()) => Ok(refusal_json(
            reason_code!("cell_not_licensed"),
            "temporal_interval.route_frozen: the historical candidate diagnostic does not authorize inference",
        )),
        Err(IoError::Refused { code, message }) => Ok(refusal_json(code, &message)),
        Err(other) => Err(crate::value_err(other.to_string())),
    }
}

/// Internal acceptance carrier; never labels an unmeasured candidate licensed.
#[cfg(feature = "calibration-internal")]
#[pyclass]
#[doc(hidden)]
struct NativeTemporalIntervalCandidate {
    wire: antecedent_io::temporal_interval_artifact::TemporalIntervalArtifactWire,
}

#[cfg(feature = "calibration-internal")]
#[pymethods]
impl NativeTemporalIntervalCandidate {
    fn payload(&self) -> PyResult<String> {
        serde_json::to_string(&self.wire).map_err(|e| crate::value_err(e.to_string()))
    }
    fn export<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.wire
            .export()
            .map(|b| PyBytes::new(py, &b))
            .map_err(|e| CausalSerializationError::new_err(e.to_string()))
    }
}

/// Actual public-route preparation, available only in internal acceptance builds.
#[cfg(feature = "calibration-internal")]
#[doc(hidden)]
#[pyfunction]
#[pyo3(signature = (snapshot_id, units, sequence, estimand, fixed_state, law, point, replicates, seed, level, method, min_units, max_failed_fraction))]
#[allow(clippy::too_many_arguments)]
fn temporal_dependent_interval_candidate(
    py: Python<'_>,
    snapshot_id: &str,
    units: Option<Vec<UnitTuple>>,
    sequence: (u32, u32),
    estimand: &str,
    fixed_state: Option<u32>,
    law: Option<LawTuple>,
    point: Option<u32>,
    replicates: usize,
    seed: u64,
    level: f64,
    method: &str,
    min_units: usize,
    max_failed_fraction: f64,
) -> Outcome<NativeTemporalIntervalCandidate> {
    // Preserve the released entry's complete argument validation and refusals.
    let refusal = temporal_dependent_interval_closed(
        snapshot_id,
        units.clone(),
        sequence,
        estimand,
        fixed_state,
        law.clone(),
        point,
        replicates,
        seed,
        level,
        method,
        min_units,
        max_failed_fraction,
    )?;
    let detail = serde_json::from_str::<serde_json::Value>(&refusal)
        .map_err(|e| crate::value_err(e.to_string()))?;
    if detail["detail"] != "temporal_interval.route_frozen" {
        return Ok((None, Some(refusal)));
    }
    let method = match method {
        "percentile" => IntervalMethod::Percentile,
        "basic" => IntervalMethod::Basic,
        "studentized" => IntervalMethod::Studentized,
        _ => return Err(crate::value_err("invalid interval method")),
    };
    let estimand = match (estimand, fixed_state) {
        ("observed_initial_state", None) => IntervalEstimand::Observed,
        ("fixed_initial_state", Some(s0)) => IntervalEstimand::FixedState(s0),
        ("marginalized_initial_state", None) => IntervalEstimand::Marginalized(
            spec_of(law, point).map_err(|e| crate::value_err(e.to_string()))?,
        ),
        _ => return Err(crate::value_err("invalid interval estimand")),
    };
    let config =
        DependentIntervalConfig { replicates, seed, level, method, min_units, max_failed_fraction };
    let ctx = crate::py_execution_context(seed, 1);
    let snapshot_id = snapshot_id.to_owned();
    crate::detach_catch(py, move || {
        produce(
            antecedent::temporal_dependent_interval_candidate(
                &snapshot_id,
                units_of(units),
                [sequence.0, sequence.1],
                estimand,
                &config,
                &ctx,
            )
            .map(|wire| NativeTemporalIntervalCandidate { wire }),
        )
    })
}

/// Measured standard route: original whole-unit source and current scalar evidence.
#[pyfunction]
#[pyo3(signature = (snapshot_id, units, sequence, estimand, fixed_state, law, point, replicates, seed, level, method, min_units, max_failed_fraction, *, memory_limit_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
fn temporal_dependent_interval_measured(
    py: Python<'_>,
    snapshot_id: &str,
    units: Option<Vec<UnitTuple>>,
    sequence: (u32, u32),
    estimand: &str,
    fixed_state: Option<u32>,
    law: Option<LawTuple>,
    point: Option<u32>,
    replicates: usize,
    seed: u64,
    level: f64,
    method: &str,
    min_units: usize,
    max_failed_fraction: f64,
    memory_limit_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> Outcome<crate::measured_inference_api::NativeMeasuredInference> {
    let numerical_bytes = units
        .as_ref()
        .map_or(0, |u| u.iter().map(|(_, h)| h.len()).sum::<usize>())
        .saturating_mul(128);
    crate::measured_inference_api::before_work(
        0,
        numerical_bytes,
        memory_limit_bytes,
        cancel.as_ref(),
    )?;
    // Preserve complete original-domain validation before scoped inference.
    let refusal = temporal_dependent_interval_closed(
        snapshot_id,
        units.clone(),
        sequence,
        estimand,
        fixed_state,
        law.clone(),
        point,
        replicates,
        seed,
        level,
        method,
        min_units,
        max_failed_fraction,
    )?;
    let detail = serde_json::from_str::<serde_json::Value>(&refusal)
        .map_err(|e| crate::value_err(e.to_string()))?;
    if detail["detail"] != "temporal_interval.route_frozen" {
        return Ok((None, Some(refusal)));
    }
    let method = match method {
        "percentile" => IntervalMethod::Percentile,
        "basic" => IntervalMethod::Basic,
        "studentized" => IntervalMethod::Studentized,
        _ => return Err(crate::value_err("invalid interval method")),
    };
    let estimand = match (estimand, fixed_state) {
        ("observed_initial_state", None) => IntervalEstimand::Observed,
        ("fixed_initial_state", Some(s0)) => IntervalEstimand::FixedState(s0),
        ("marginalized_initial_state", None) => IntervalEstimand::Marginalized(
            spec_of(law, point).map_err(|e| crate::value_err(e.to_string()))?,
        ),
        _ => return Err(crate::value_err("invalid interval estimand")),
    };
    let config =
        DependentIntervalConfig { replicates, seed, level, method, min_units, max_failed_fraction };
    let ctx = crate::measured_inference_api::context(seed, memory_limit_bytes, cancel);
    let snapshot_id = snapshot_id.to_owned();
    crate::detach_catch(py, move || {
        let source = antecedent::temporal_dependent_interval_candidate(
            &snapshot_id,
            units_of(units),
            [sequence.0, sequence.1],
            estimand,
            &config,
            &ctx,
        );
        let source = match source {
            Ok(wire) => wire,
            Err(cause) => return produce(Err(cause)),
        };
        let bytes = source.export().map_err(crate::transport_common::error)?;
        let measured =
            antecedent_io::temporal_interval_artifact::TemporalIntervalArtifactWire::measure(
                &bytes,
                &source.seal,
                &antecedent_io::temporal_interval_artifact::TemporalIntervalConsumeLimits::default(
                ),
                &ctx,
            );
        let measured = match measured {
            Ok(value) => value,
            Err(cause) => return produce(Err(cause)),
        };
        crate::measured_inference_api::check_cancelled(&ctx)?;
        Ok((Some(crate::measured_inference_api::NativeMeasuredInference::from_io(measured)?), None))
    })
}

/// Fresh internal consumer reruns the original whole-unit estimator/resampling.
#[cfg(feature = "calibration-internal")]
#[doc(hidden)]
#[pyfunction]
fn consume_temporal_interval_candidate(
    py: Python<'_>,
    artifact: &[u8],
    expected_identity: &str,
) -> Outcome<NativeTemporalIntervalCandidate> {
    use antecedent_io::temporal_interval_artifact::{
        TemporalIntervalArtifactWire, TemporalIntervalConsumeLimits,
    };
    let artifact = artifact.to_vec();
    let expected_identity = expected_identity.to_owned();
    crate::detach_catch(py, move || {
        let outcome = (|| {
            let wire = TemporalIntervalArtifactWire::decode(&artifact)?;
            if wire.seal != expected_identity {
                return Err(IoError::Refused {
                code: reason_code!("invalid_argument"),
                message: "temporal_interval_artifact.expected_identity_mismatch: retained identity differs".into(),
            });
            }
            let config = wire.config.to_config()?;
            let ctx = crate::py_execution_context(config.seed, 1);
            let (wire, _) = TemporalIntervalArtifactWire::consume(
                &artifact,
                &TemporalIntervalConsumeLimits::default(),
                &ctx,
            )?;
            Ok(NativeTemporalIntervalCandidate { wire })
        })();
        consume(outcome)
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(temporal_dependent_interval_measured, module)?)?;
    #[cfg(feature = "calibration-internal")]
    {
        module.add_class::<NativeTemporalIntervalCandidate>()?;
        module.add_function(wrap_pyfunction!(temporal_dependent_interval_candidate, module)?)?;
        module.add_function(wrap_pyfunction!(consume_temporal_interval_candidate, module)?)?;
    }
    module.add_class::<NativeTemporalInitialState>()?;
    module.add_function(wrap_pyfunction!(temporal_initial_state_prepare, module)?)?;
    module.add_function(wrap_pyfunction!(consume_temporal_initial_state_artifact, module)?)?;
    module.add_function(wrap_pyfunction!(consume_temporal_refresh_artifact, module)?)?;
    module.add_function(wrap_pyfunction!(temporal_dependent_interval_closed, module)?)?;
    Ok(())
}
