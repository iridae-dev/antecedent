//! Python bindings for an autoregressive mechanism refreshed across observation periods,
//! least squares and the conjugate posterior on the same rows (an unlicensed, uncalibrated
//! route: the result is the posterior of the stated linear Gaussian model under the stated
//! prior, and a prior alone identifies no causal target).
//!
//! Python builds the declarations; every identity, check and refusal rule is Rust's. A refusal
//! comes back as structured JSON for the Python layer to raise as its own exception type.
use crate::bridge::RefusalJson;
use antecedent::consume_period_refresh_artifact as consume_ledger;
use antecedent::period_refresh::{
    PeriodRefresh, PeriodRefreshOutcome, PeriodRefreshSpec, SummaryOnlyFit,
};
use antecedent_core::reason_code;
use antecedent_estimate::temporal_refresh::{
    MechanismStanding, ObservationCadence, ObservationPeriod, TemporalWindowIdentity,
};
use antecedent_io::IoError;
use antecedent_io::period_refresh_artifact::{PriorWire, product_label};
use antecedent_state::{ConjugateOlsPrior, ConjugateOlsSummary, OlsSnapshot, StateError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

const LEDGER_PREFIX: &[u8] = b"ANTECEDENT-PERIOD-REFRESH\x01";
const SUMMARY_PREFIX: &[u8] = b"ANTECEDENT-PERIOD-REFRESH-SUMMARY\x01";

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
type Refusable<T> = (Option<T>, Option<String>);
/// A payload or a structured refusal, or a raised error.
type Outcome<T> = PyResult<Refusable<T>>;

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

fn refuse<T>(detail: &str, message: &str) -> Refusable<T> {
    (None, Some(refusal_json(reason_code!("invalid_argument"), &format!("{detail}: {message}"))))
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

fn outcome_value(outcome: &PeriodRefreshOutcome) -> serde_json::Value {
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
}

fn outcome_json(outcome: &PeriodRefreshOutcome) -> String {
    outcome_value(outcome).to_string()
}

fn consume_outcome<T>(result: Result<T, IoError>) -> Outcome<T> {
    match result {
        Ok(value) => Ok((Some(value), None)),
        Err(IoError::Refused { code, message }) => Ok((None, Some(refusal_json(code, &message)))),
        Err(other) => Err(crate::serialization_err(other.to_string())),
    }
}

fn frame(raw: &[u8]) -> Vec<u8> {
    framed_with(LEDGER_PREFIX, raw)
}

fn framed_with(prefix: &[u8], raw: &[u8]) -> Vec<u8> {
    let mut framed = prefix.to_vec();
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

/// Summaries resolved at explicit execution boundaries. A failed posterior summary
/// remains a read refusal, rather than reporting an already-applied mutation as failed.
struct PeriodReadState {
    posterior: Result<ConjugateOlsSummary, StateError>,
    least_squares: OlsSnapshot,
    prior_digest: String,
    native_prior_digest: String,
}

impl PeriodReadState {
    fn capture(refresh: &PeriodRefresh) -> Self {
        let prior = refresh.prior();
        Self {
            posterior: refresh.summary(),
            least_squares: refresh.fit().snapshot(),
            native_prior_digest: prior.digest(),
            prior_digest: PriorWire {
                mean: prior.mean().to_vec(),
                precision: prior.precision().to_vec(),
                shape: prior.shape(),
                rate: prior.rate(),
            }
            .digest(),
        }
    }

    fn posterior_json(&self) -> Refusable<String> {
        match &self.posterior {
            Ok(summary) => (
                Some(
                    serde_json::json!({
                        "n": summary.n,
                        "mean": summary.mean,
                        "precision": summary.precision,
                        "shape": summary.shape,
                        "rate": summary.rate,
                    })
                    .to_string(),
                ),
                None,
            ),
            Err(error) => (None, Some(state_refusal(error, "period_refresh.state"))),
        }
    }

    fn least_squares_json(&self) -> String {
        let fit = &self.least_squares;
        serde_json::json!({
            "n": fit.n,
            "beta": fit.beta,
            "residual_variance": fit.residual_variance,
        })
        .to_string()
    }
}

fn events_json(refresh: &PeriodRefresh) -> String {
    serde_json::Value::Array(
        refresh
            .events()
            .iter()
            .map(|event| {
                let mut value = event.outcome.as_ref().map_or_else(
                || serde_json::json!({"outcome": "start", "contiguous": false, "invalidated": []}),
                outcome_value,
            );
                value["snapshot_id"] = serde_json::json!(event.window.snapshot_id);
                value["period"] =
                    serde_json::json!([event.window.period.start, event.window.period.end]);
                value
            })
            .collect::<Vec<_>>(),
    )
    .to_string()
}

/// An autoregressive mechanism kept across observation periods.
#[pyclass(skip_from_py_object)]
struct NativePeriodRefresh {
    inner: PeriodRefresh,
    read_state: PeriodReadState,
}

impl NativePeriodRefresh {
    fn from_inner(inner: PeriodRefresh) -> Self {
        let read_state = PeriodReadState::capture(&inner);
        Self { inner, read_state }
    }

    fn refresh_read_state(&mut self) {
        self.read_state = PeriodReadState::capture(&self.inner);
    }
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
    ) -> Refusable<Self> {
        let Some(cadence) = ObservationCadence::new(cadence_tick) else {
            return refuse("period_refresh.invalid_cadence", "the cadence tick must be positive");
        };
        let prior = match prior_of(prior) {
            Ok(prior) => prior,
            Err(error) => return (None, Some(state_refusal(&error, "period_refresh.prior"))),
        };
        let standing = if invariant {
            MechanismStanding::DeclaredInvariant
        } else {
            MechanismStanding::Undeclared
        };
        let spec = PeriodRefreshSpec { series_id, cadence, initial_law, standing, lags };
        match PeriodRefresh::start(&spec, &identity_of(window), &prior, &values) {
            Ok(inner) => (Some(Self::from_inner(inner)), None),
            Err(error) => (None, Some(state_refusal(&error, "period_refresh.start"))),
        }
    }

    /// Apply a later observation period. A refused period changes nothing.
    fn refresh(
        &mut self,
        window: WindowTuple,
        cadence_tick: u64,
        initial_law: &str,
        values: Vec<f64>,
    ) -> Refusable<String> {
        let Some(cadence) = ObservationCadence::new(cadence_tick) else {
            return refuse("period_refresh.invalid_cadence", "the cadence tick must be positive");
        };
        match self.inner.refresh(&identity_of(window), cadence, initial_law, &values) {
            Ok(outcome) => {
                self.refresh_read_state();
                (Some(outcome_json(&outcome)), None)
            }
            Err(error) => (None, Some(state_refusal(&error, "period_refresh.refused"))),
        }
    }

    /// The cached posterior, without inversion or retained-row replay on reads.
    fn posterior_json(&self) -> Refusable<String> {
        self.read_state.posterior_json()
    }

    /// The cached least-squares summary; singular designs have no coefficients.
    fn least_squares_json(&self) -> String {
        self.read_state.least_squares_json()
    }

    /// Explicitly materialize event history without resolving numerical summaries.
    fn events_json(&self) -> String {
        events_json(&self.inner)
    }

    /// Cached canonical wire identity of the stated prior.
    fn prior_digest(&self) -> String {
        self.read_state.prior_digest.clone()
    }

    /// The retained ledger artifact, framed.
    fn export_ledger(&self, py: Python<'_>) -> Refusable<Py<PyBytes>> {
        match self.inner.export_ledger() {
            Ok(raw) => (Some(PyBytes::new(py, &frame(&raw)).unbind()), None),
            Err(error) => (None, Some(state_refusal(&error, "period_refresh.ledger"))),
        }
    }

    /// The summary-only artifact of the least-squares fit, framed. This fit keeps its rows.
    fn export_summary_only(&self, py: Python<'_>) -> Refusable<Py<PyBytes>> {
        match self.inner.export_summary_only() {
            Ok(raw) => (Some(PyBytes::new(py, &framed_with(SUMMARY_PREFIX, &raw)).unbind()), None),
            Err(error) => (None, Some(state_refusal(&error, "period_refresh.summary_only"))),
        }
    }

    /// Remove one retained row; both summaries are rebuilt from the rows that remain.
    fn evict_row(&mut self, id: u64) -> Refusable<String> {
        match self.inner.evict_row(id) {
            Ok(outcome) => {
                self.refresh_read_state();
                (Some(format!("{outcome:?}").to_lowercase()), None)
            }
            Err(error) => (None, Some(state_refusal(&error, "period_refresh.evict"))),
        }
    }

    /// The posterior under `prior`; any prior other than the stated one is refused.
    fn summary_under(&self, prior: PriorTuple) -> Refusable<String> {
        let prior = match prior_of(prior) {
            Ok(prior) => prior,
            Err(error) => return (None, Some(state_refusal(&error, "period_refresh.prior"))),
        };
        if prior.digest() != self.read_state.native_prior_digest {
            return (None, Some(state_refusal(
                &StateError::StaleCache(
                    "the posterior was built under another prior; a different prior is a different \
                     posterior, so rebuild from every retained block under the new prior"
                        .into(),
                ),
                "period_refresh.prior_changed",
            )));
        }
        match &self.read_state.posterior {
            Ok(summary) => (Some(serde_json::json!({ "n": summary.n }).to_string()), None),
            Err(error) => (None, Some(state_refusal(error, "period_refresh.prior_changed"))),
        }
    }

    /// Rebuild the posterior from every retained block under a new prior.
    fn reprior(&mut self, prior: PriorTuple) -> Refusable<bool> {
        let prior = match prior_of(prior) {
            Ok(prior) => prior,
            Err(error) => return (None, Some(state_refusal(&error, "period_refresh.prior"))),
        };
        match self.inner.reprior(&prior) {
            Ok(()) => {
                self.refresh_read_state();
                (Some(true), None)
            }
            Err(error) => (None, Some(state_refusal(&error, "period_refresh.prior"))),
        }
    }

    /// Compare the cached results with a fresh rebuild from the retained rows.
    fn check_against_fresh(&self) -> Refusable<bool> {
        match self.inner.check_against_fresh() {
            Ok(()) => (Some(true), None),
            Err(error) => (None, Some(state_refusal(&error, "period_refresh.stale"))),
        }
    }
}

/// A least-squares summary read from its summary-only artifact: it serves the summary and
/// refuses every operation that needs rows.
#[pyclass(skip_from_py_object)]
struct NativePeriodRefreshSummary {
    inner: SummaryOnlyFit,
}

#[pymethods]
impl NativePeriodRefreshSummary {
    /// Decode and verify a framed summary-only artifact.
    #[staticmethod]
    fn consume(artifact: &[u8]) -> Refusable<Self> {
        let Some(raw) = artifact.strip_prefix(SUMMARY_PREFIX) else {
            return refuse(
                "period_refresh.summary_only",
                "this is not a summary-only period refresh artifact; a full ledger is read with \
                 the ledger consumer",
            );
        };
        match SummaryOnlyFit::consume(raw) {
            Ok(inner) => (Some(Self { inner }), None),
            Err(error) => (None, Some(state_refusal(&error, "period_refresh.summary_only"))),
        }
    }

    /// The retained summary as JSON.
    fn payload(&self) -> String {
        let ledger = self.inner.ledger();
        serde_json::json!({
            "retention": self.inner.retention().as_str(),
            "source": self.inner.source(),
            "n": self.inner.n(),
            "beta": self.inner.beta(),
            "residual_variance": self.inner.residual_variance(),
            "source_digest": self.inner.source_digest(),
            "ledger": {
                "rows": ledger.rows,
                "live_rows": ledger.live_rows,
                "effective_n": ledger.effective_n,
            },
        })
        .to_string()
    }

    /// Refit from retained rows; always refused.
    fn refit_from_rows(&self) -> Refusable<bool> {
        refused_operation(self.inner.refit_from_rows())
    }

    /// Evict one row; always refused.
    fn evict_row(&self, id: u64) -> Refusable<bool> {
        refused_operation(self.inner.evict_row(id))
    }

    /// Change a mask; always refused.
    fn set_mask(&self, id: u64, masked: bool) -> Refusable<bool> {
        refused_operation(self.inner.set_mask(id, masked))
    }

    /// Replay the rows of the summary; always refused.
    fn replay_rows(&self) -> Refusable<bool> {
        refused_operation(self.inner.replay_rows())
    }
}

fn refused_operation(result: Result<(), StateError>) -> Refusable<bool> {
    match result {
        Ok(()) => (Some(true), None),
        Err(error) => (None, Some(state_refusal(&error, "period_refresh.retention"))),
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
    module.add_class::<NativePeriodRefreshSummary>()?;
    module.add_function(wrap_pyfunction!(consume_period_refresh_artifact, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prior() -> PriorTuple {
        (vec![0.0; 2], vec![1.0, 0.0, 0.0, 1.0], 3.0, 2.0)
    }

    fn window(start: i64, end: i64, snapshot: &str) -> WindowTuple {
        (
            "graph".into(),
            2,
            vec![("v".into(), 1)],
            vec![],
            vec![],
            vec!["observational".into()],
            start,
            end,
            vec!["unit".into()],
            snapshot.into(),
            "proof".into(),
        )
    }

    fn session() -> NativePeriodRefresh {
        let (session, refusal) = NativePeriodRefresh::start(
            "v",
            1,
            true,
            prior(),
            window(0, 3, "first"),
            1,
            "law",
            vec![1.0, 2.0, 4.0],
        );
        assert!(refusal.is_none(), "{refusal:?}");
        session.expect("valid first period")
    }

    #[test]
    fn cached_reads_do_not_resolve_or_replay_the_inner_session() {
        let mut session = session();
        let before = session.posterior_json();
        let least_squares = session.least_squares_json();
        // Deliberately bypass the binding's mutation boundary. A read that consulted
        // the live numerical state would observe five rows instead of the cached two.
        session
            .inner
            .refresh(
                &identity_of(window(3, 6, "second")),
                ObservationCadence::new(1).expect("positive cadence"),
                "law",
                &[8.0, 16.0, 32.0],
            )
            .expect("compatible next period");
        assert_eq!(session.posterior_json(), before);
        assert_eq!(session.least_squares_json(), least_squares);
        let events: serde_json::Value = serde_json::from_str(&session.events_json()).unwrap();
        assert_eq!(events.as_array().unwrap().len(), 2);
        session.refresh_read_state();
        let posterior: serde_json::Value =
            serde_json::from_str(&session.posterior_json().0.unwrap()).unwrap();
        assert_eq!(posterior["n"], 5);
    }

    #[test]
    fn successful_mutations_refresh_reads_and_refusals_preserve_them() {
        let mut session = session();
        assert!(
            session.refresh(window(3, 6, "second"), 1, "law", vec![8.0, 16.0, 32.0]).1.is_none()
        );
        let advanced = session.posterior_json();
        assert!(
            session.refresh(window(3, 6, "second"), 1, "law", vec![8.0, 16.0, 32.0]).1.is_some()
        );
        assert_eq!(session.posterior_json(), advanced);
        assert!(session.evict_row(1).1.is_none());
        let evicted: serde_json::Value =
            serde_json::from_str(&session.least_squares_json()).unwrap();
        assert_eq!(evicted["n"], 4);
        let old_digest = session.prior_digest();
        let mut changed = prior();
        changed.0 = vec![1.0, 1.0];
        assert!(session.reprior(changed.clone()).1.is_none());
        assert_ne!(session.prior_digest(), old_digest);
        assert!(session.summary_under(prior()).1.is_some());
        assert!(session.summary_under(changed).1.is_none());
    }

    #[test]
    fn posterior_failure_is_retained_without_hiding_other_read_models() {
        let mut session = session();
        session.read_state.posterior = Err(StateError::Numerical("posterior read failure".into()));
        let (posterior, refusal) = session.posterior_json();
        assert!(posterior.is_none());
        assert!(refusal.unwrap().contains("posterior read failure"));
        assert!(session.summary_under(prior()).1.unwrap().contains("posterior read failure"));
        let fit: serde_json::Value = serde_json::from_str(&session.least_squares_json()).unwrap();
        assert_eq!(fit["n"], 2);
        assert_eq!(session.prior_digest().len(), 64);
        let events: serde_json::Value = serde_json::from_str(&session.events_json()).unwrap();
        assert_eq!(events.as_array().unwrap().len(), 1);
    }
}
