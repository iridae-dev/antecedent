//! Python bindings for an autoregressive mechanism refreshed across observation periods,
//! least squares and the conjugate posterior on the same rows (an unlicensed, uncalibrated
//! route: the result is the posterior of the stated linear Gaussian model under the stated
//! prior, and a prior alone identifies no causal target).
//!
//! Python builds the declarations; every identity, check and refusal rule is Rust's. A refusal
//! comes back as structured JSON for the Python layer to raise as its own exception type.
use crate::bridge::RefusalJson;
use antecedent::consume_period_refresh_artifact as consume_ledger;
use antecedent::period_refresh::{PeriodRefresh, PeriodRefreshOutcome, PeriodRefreshSpec};
use antecedent_core::reason_code;
use antecedent_estimate::temporal_refresh::{
    MechanismStanding, ObservationCadence, ObservationPeriod, TemporalWindowIdentity,
};
use antecedent_io::IoError;
use antecedent_io::period_refresh_artifact::product_label;
use antecedent_state::{ConjugateOlsPrior, StateError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

const LEDGER_PREFIX: &[u8] = b"ANTECEDENT-PERIOD-REFRESH\x01";

/// `(mean, precision, shape, rate)`.
type PriorTuple = (Vec<f64>, Vec<f64>, f64, f64);
/// `(graph, horizon, lag_alignment, intervention_history, selection_targets, regimes, start,
/// end, unit_ids, snapshot_id, proof_id)`.
type WindowTuple = (
    String,
    usize,
    Vec<(String, u8)>,
    Vec<String>,
    Vec<String>,
    Vec<String>,
    i64,
    i64,
    Vec<String>,
    String,
    String,
);
/// A payload or a structured refusal.
type Outcome<T> = PyResult<(Option<T>, Option<String>)>;

fn refusal_json(code: &str, message: &str) -> String {
    RefusalJson::from_coded_message(code, "period_refresh", message).json()
}

/// A state refusal as structured JSON; `detail` names the slot unless the message already does.
fn state_refusal(error: &StateError, detail: &str) -> String {
    let message = match error {
        StateError::InvalidEvent(m)
        | StateError::StaleCache(m)
        | StateError::Shape(m)
        | StateError::Numerical(m)
        | StateError::UnknownId(m) => m.clone(),
        other => other.to_string(),
    };
    let message = if message.starts_with("period_refresh.") {
        message
    } else {
        format!("{detail}: {message}")
    };
    refusal_json(reason_code!("invalid_argument"), &message)
}

fn refuse<T>(detail: &str, message: &str) -> Outcome<T> {
    Ok((
        None,
        Some(refusal_json(reason_code!("invalid_argument"), &format!("{detail}: {message}"))),
    ))
}

fn identity_of(window: WindowTuple) -> TemporalWindowIdentity {
    let (
        graph_id,
        horizon,
        lag_alignment,
        history,
        targets,
        regimes,
        start,
        end,
        units,
        snapshot,
        proof,
    ) = window;
    TemporalWindowIdentity {
        graph_id,
        horizon,
        lag_alignment,
        intervention_history: history,
        selection_targets: targets.into_iter().collect(),
        regimes: regimes.into_iter().collect(),
        period: ObservationPeriod { start, end },
        unit_ids: units.into_iter().collect(),
        snapshot_id: snapshot,
        proof_id: proof,
    }
}

fn prior_of(prior: PriorTuple) -> Result<ConjugateOlsPrior, StateError> {
    let (mean, precision, shape, rate) = prior;
    ConjugateOlsPrior::new(mean, precision, shape, rate)
}

fn outcome_json(outcome: &PeriodRefreshOutcome) -> String {
    match outcome {
        PeriodRefreshOutcome::Extended { contiguous } => {
            serde_json::json!({ "outcome": "extended", "contiguous": contiguous, "invalidated": [] })
        }
        PeriodRefreshOutcome::Restarted { invalidated } => serde_json::json!({
            "outcome": "restarted",
            "contiguous": false,
            "invalidated": invalidated.iter().map(|p| product_label(*p)).collect::<Vec<_>>(),
        }),
    }
    .to_string()
}

fn consume_outcome<T>(result: Result<T, IoError>) -> Outcome<T> {
    match result {
        Ok(value) => Ok((Some(value), None)),
        Err(IoError::Refused { code, message }) => Ok((None, Some(refusal_json(code, &message)))),
        Err(other) => Err(crate::serialization_err(other.to_string())),
    }
}

fn frame(raw: &[u8]) -> Vec<u8> {
    let mut framed = LEDGER_PREFIX.to_vec();
    framed.extend_from_slice(raw);
    framed
}

/// The retained ledger as JSON: events, live blocks (snapshot ids and row counts), results.
fn ledger_json(wire: &antecedent_io::period_refresh_artifact::PeriodRefreshArtifactWire) -> String {
    serde_json::json!({
        "series_id": wire.series_id,
        "lags": wire.lags,
        "standing": wire.standing,
        "prior_digest": wire.prior_digest,
        "rows_used": wire.rows_used,
        "events": wire.events.iter().map(|e| serde_json::json!({
            "snapshot_id": e.window.snapshot_id,
            "period": [e.window.period_start, e.window.period_end],
            "outcome": e.outcome,
            "contiguous": e.contiguous,
            "invalidated": e.invalidated,
        })).collect::<Vec<_>>(),
        "blocks": wire.blocks.iter().map(|b| serde_json::json!({
            "snapshot_id": b.snapshot_id,
            "rows": b.rows.len(),
            "digest": b.digest,
        })).collect::<Vec<_>>(),
        "posterior": {
            "n": wire.posterior.n,
            "mean": wire.posterior.mean,
            "precision": wire.posterior.precision,
            "shape": wire.posterior.shape,
            "rate": wire.posterior.rate,
        },
        "least_squares": {
            "n": wire.least_squares.n,
            "beta": wire.least_squares.beta,
            "residual_variance": wire.least_squares.residual_variance,
        },
    })
    .to_string()
}

/// An autoregressive mechanism kept across observation periods.
#[pyclass(skip_from_py_object)]
struct NativePeriodRefresh {
    inner: PeriodRefresh,
}

#[pymethods]
impl NativePeriodRefresh {
    /// Fit the first period.
    #[staticmethod]
    #[pyo3(signature = (series_id, lags, invariant, prior, window, cadence_tick, initial_law, values))]
    #[allow(clippy::too_many_arguments)]
    fn start(
        series_id: &str,
        lags: usize,
        invariant: bool,
        prior: PriorTuple,
        window: WindowTuple,
        cadence_tick: u64,
        initial_law: &str,
        values: Vec<f64>,
    ) -> Outcome<Self> {
        let Some(cadence) = ObservationCadence::new(cadence_tick) else {
            return refuse("period_refresh.invalid_cadence", "the cadence tick must be positive");
        };
        let prior = match prior_of(prior) {
            Ok(prior) => prior,
            Err(error) => return Ok((None, Some(state_refusal(&error, "period_refresh.prior")))),
        };
        let standing = if invariant {
            MechanismStanding::DeclaredInvariant
        } else {
            MechanismStanding::Undeclared
        };
        let spec = PeriodRefreshSpec { series_id, cadence, initial_law, standing, lags };
        match PeriodRefresh::start(&spec, &identity_of(window), &prior, &values) {
            Ok(inner) => Ok((Some(Self { inner }), None)),
            Err(error) => Ok((None, Some(state_refusal(&error, "period_refresh.start")))),
        }
    }

    /// Apply a later observation period. A refused period changes nothing.
    fn refresh(
        &mut self,
        window: WindowTuple,
        cadence_tick: u64,
        initial_law: &str,
        values: Vec<f64>,
    ) -> Outcome<String> {
        let Some(cadence) = ObservationCadence::new(cadence_tick) else {
            return refuse("period_refresh.invalid_cadence", "the cadence tick must be positive");
        };
        match self.inner.refresh(&identity_of(window), cadence, initial_law, &values) {
            Ok(outcome) => Ok((Some(outcome_json(&outcome)), None)),
            Err(error) => Ok((None, Some(state_refusal(&error, "period_refresh.refused")))),
        }
    }

    /// The ledger and the cached results as JSON.
    fn payload(&self) -> Outcome<String> {
        match self.inner.ledger_wire() {
            Ok(wire) => Ok((Some(ledger_json(&wire)), None)),
            Err(error) => Ok((None, Some(state_refusal(&error, "period_refresh.ledger")))),
        }
    }

    /// The retained ledger artifact, framed.
    fn export_ledger(&self, py: Python<'_>) -> Outcome<Py<PyBytes>> {
        match self.inner.export_ledger() {
            Ok(raw) => Ok((Some(PyBytes::new(py, &frame(&raw)).unbind()), None)),
            Err(error) => Ok((None, Some(state_refusal(&error, "period_refresh.ledger")))),
        }
    }

    /// Remove one retained row; both summaries are rebuilt from the rows that remain.
    fn evict_row(&mut self, id: u64) -> Outcome<String> {
        match self.inner.evict_row(id) {
            Ok(outcome) => Ok((Some(format!("{outcome:?}").to_lowercase()), None)),
            Err(error) => Ok((None, Some(state_refusal(&error, "period_refresh.evict")))),
        }
    }

    /// The posterior under `prior`; any prior other than the stated one is refused.
    fn summary_under(&self, prior: PriorTuple) -> Outcome<String> {
        let prior = match prior_of(prior) {
            Ok(prior) => prior,
            Err(error) => return Ok((None, Some(state_refusal(&error, "period_refresh.prior")))),
        };
        match self.inner.summary_under(&prior) {
            Ok(summary) => Ok((Some(serde_json::json!({ "n": summary.n }).to_string()), None)),
            Err(error) => Ok((None, Some(state_refusal(&error, "period_refresh.prior_changed")))),
        }
    }

    /// Rebuild the posterior from every retained block under a new prior.
    fn reprior(&mut self, prior: PriorTuple) -> Outcome<bool> {
        let prior = match prior_of(prior) {
            Ok(prior) => prior,
            Err(error) => return Ok((None, Some(state_refusal(&error, "period_refresh.prior")))),
        };
        match self.inner.reprior(&prior) {
            Ok(()) => Ok((Some(true), None)),
            Err(error) => Ok((None, Some(state_refusal(&error, "period_refresh.prior")))),
        }
    }

    /// Compare the cached results with a fresh rebuild from the retained rows.
    fn check_against_fresh(&self) -> Outcome<bool> {
        match self.inner.check_against_fresh() {
            Ok(()) => Ok((Some(true), None)),
            Err(error) => Ok((None, Some(state_refusal(&error, "period_refresh.stale")))),
        }
    }
}

/// Independently recompute a framed ledger artifact from its retained blocks.
#[pyfunction]
fn consume_period_refresh_artifact(artifact: &[u8]) -> Outcome<String> {
    let Some(raw) = artifact.strip_prefix(LEDGER_PREFIX) else {
        return Err(crate::serialization_err("invalid period refresh artifact format"));
    };
    consume_outcome(consume_ledger(raw).map(|(wire, replay)| {
        let mut payload: serde_json::Value =
            serde_json::from_str(&ledger_json(&wire)).unwrap_or(serde_json::Value::Null);
        payload["recomputed"] = serde_json::json!({
            "posterior": {
                "n": replay.posterior.n,
                "mean": replay.posterior.mean,
                "precision": replay.posterior.precision,
                "shape": replay.posterior.shape,
                "rate": replay.posterior.rate,
            },
            "least_squares": {
                "n": replay.least_squares.n,
                "beta": replay.least_squares.beta,
                "residual_variance": replay.least_squares.residual_variance,
            },
            "live_snapshots": replay.live_snapshots,
        });
        payload.to_string()
    }))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativePeriodRefresh>()?;
    module.add_function(wrap_pyfunction!(consume_period_refresh_artifact, module)?)?;
    Ok(())
}
