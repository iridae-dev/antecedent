//! Bounded Python bridge for the cell-AIPW recalculation session and portable score resume
//! (2.3 C2 remainder).
//!
//! Rust owns the stage model, the planner, the instrumented executors, the receipt counts, the
//! frozen score artifact and every refusal. Python hands over numpy columns and reads back JSON
//! and artifact bytes. Every fallible call returns `(value, refusal_json)`: a refused plan, a
//! request that needs data or a fit, a malformed run-time request and a changed artifact
//! identity come back as structured JSON for the Python layer to raise; corruption and
//! unsupported versions raise `CausalSerializationError`.
//!
//! Three handles:
//!
//! - `CellSessionHandle`: the cell-AIPW route (discrete joint binary treatments).
//! - `CrossfitSessionHandle`: the cross-fitted AIPW route, so its frozen scores can be exported.
//! - `ScoreResumeHandle`: built only from artifact bytes; one licensed operation, a retarget of
//!   the frozen scores, with zero fits.
#![allow(clippy::too_many_arguments)]

use std::collections::BTreeMap;

use antecedent::analysis::recalc_cell::{
    CellRequest, CellSession, CellSpec, ScoreQuantity, ScoreResumeRequest, ScoreResumeSession,
    execute_cell_with_receipt, execute_resumed_retarget, freeze_crossfit_scores,
};
use antecedent::analysis::recalc_receipt::{
    RecalcOutcome, RecalcRequest, RecalcRunError, RecalcSession, StageCounts, TargetWeights,
    UtilitySpec, execute_with_receipt,
};
use antecedent_core::recalc::{
    Boundary, MissingDependency, RecalcCapabilities, RecalcPlan, RefusalReason, RequestSupport,
    ResumeContext, RetargetSupport, Stage, StageIdentities, StageIdentity,
};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_estimate::{AipwAte, CellSaturatedAipw, ScoreTable};
use antecedent_io::frozen_scores_artifact::{FrozenScoreTable, FrozenScoresArtifactError};
use antecedent_io::recalc_receipt_artifact::{
    CapabilitiesWire, CountsWire, RecalcReceiptArtifact, ReceiptEntryWire, identities_to_wire,
    plan_to_wire, stage_from_label,
};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::{CausalSerializationError, detach_catch, py_err, py_msg, value_err, with_reason_code};

/// `(result json, receipt artifact bytes, refusal json)`.
type RunPayload = (Option<String>, Option<Vec<u8>>, Option<String>);
/// `(result json, refusal json)`.
type ResumePayload = (Option<String>, Option<String>);
/// `((identity, artifact bytes), refusal json)`.
type ExportPayload = (Option<(String, Vec<u8>)>, Option<String>);
type Columns = Vec<(String, Vec<f64>)>;

fn invalid(detail: &str, text: impl AsRef<str>) -> PyErr {
    with_reason_code(
        value_err(format!("{detail}: {}", text.as_ref())),
        antecedent_core::reason_code!("invalid_argument"),
    )
}

// -- refusals -----------------------------------------------------------------------------

fn refusal_json(
    code: &str,
    stage: &str,
    detail: &str,
    offending: Option<&str>,
    remedy: Option<String>,
    message: &str,
    plan: Option<&RecalcPlan>,
) -> String {
    serde_json::json!({
        "code": code,
        "stage": stage,
        "detail": detail,
        "offending": offending,
        "expected": serde_json::Value::Null,
        "supplied": serde_json::Value::Null,
        "remedy": remedy,
        "message": message,
        "plan": plan.map(plan_to_wire),
    })
    .to_string()
}

/// The refusal a refused plan carries: its first refused stage and reason.
fn plan_refusal_json(plan: &RecalcPlan) -> String {
    let Some((stage, reason)) = plan.first_refusal() else {
        return refusal_json(
            antecedent_core::reason_code!("not_executed"),
            "",
            "recalc.refused",
            None,
            None,
            "recalculation refused",
            Some(plan),
        );
    };
    let offending = match reason {
        RefusalReason::Blocked { by } => Some(by.label()),
        _ => None,
    };
    let remedy = match reason {
        RefusalReason::OffGrid { licensed_route: Some(route) }
        | RefusalReason::Unsupported { licensed_route: Some(route) } => {
            Some(format!("use the separately licensed route `{route}`"))
        }
        RefusalReason::Unavailable { missing } => Some(match missing {
            MissingDependency::Fit => {
                "supply a portable fit or scores, or the data snapshot to refit".to_owned()
            }
            MissingDependency::Data => {
                "run the changed request on a live session that holds the data".to_owned()
            }
            MissingDependency::Provider => "supply a compatible provider callback".to_owned(),
        }),
        _ => None,
    };
    refusal_json(
        reason.reason_code(),
        &stage.label(),
        reason.detail(),
        offending.as_deref(),
        remedy,
        &format!("recalculation refused at {stage}: {reason}"),
        Some(plan),
    )
}

/// A malformed run-time request (`recalc.<snake_case>` detail).
fn request_refusal_json(detail: &str) -> String {
    refusal_json(
        antecedent_core::reason_code!("invalid_argument"),
        "",
        detail,
        None,
        None,
        &format!("the request is malformed: {detail}"),
        None,
    )
}

fn no_live_state_json(stage: Stage, planned: Option<&RecalcPlan>) -> String {
    refusal_json(
        antecedent_core::reason_code!("score_table_unavailable"),
        &stage.label(),
        "recalc.no_live_state",
        Some(&stage.label()),
        Some(
            "run the stage in this process, or resume with supplied data so it is recomputed \
             instead of reused"
                .to_owned(),
        ),
        &format!("no live artifact to reuse for stage {stage}"),
        planned,
    )
}

/// What one detached run produced.
enum Ran {
    Done(Box<RecalcOutcome>),
    Refused(String),
    Failed(PyErr),
}

fn classify(result: Result<RecalcOutcome, RecalcRunError>, planned: Option<&RecalcPlan>) -> Ran {
    match result {
        Ok(outcome) => Ran::Done(Box::new(outcome)),
        Err(RecalcRunError::Refused(plan)) => Ran::Refused(plan_refusal_json(&plan)),
        Err(RecalcRunError::NoLiveState(stage)) => Ran::Refused(no_live_state_json(stage, planned)),
        Err(RecalcRunError::Request(detail)) => Ran::Refused(request_refusal_json(detail)),
        Err(RecalcRunError::Execution(error)) => Ran::Failed(py_err(error)),
        Err(RecalcRunError::Receipt(error)) => Ran::Failed(py_msg(error)),
    }
}

fn export_refusal(error: RecalcRunError) -> PyResult<ExportPayload> {
    match error {
        RecalcRunError::Refused(plan) => Ok((None, Some(plan_refusal_json(&plan)))),
        RecalcRunError::NoLiveState(stage) => Ok((None, Some(no_live_state_json(stage, None)))),
        RecalcRunError::Request(detail) => Ok((None, Some(request_refusal_json(detail)))),
        RecalcRunError::Execution(error) => Err(py_err(error)),
        RecalcRunError::Receipt(error) => Err(py_msg(error)),
    }
}

fn artifact_error(error: &FrozenScoresArtifactError) -> PyErr {
    match error {
        FrozenScoresArtifactError::LimitsExceeded(what) => invalid("recalc.limits_exceeded", what),
        other => CausalSerializationError::new_err(other.to_string()),
    }
}

fn artifact_refusal(error: &FrozenScoresArtifactError) -> Option<String> {
    let (code, detail, text) = error.refusal()?;
    Some(refusal_json(code, "score_artifact", detail, Some("identity"), None, &text, None))
}

// -- request assembly ---------------------------------------------------------------------

fn retarget_of(name: &str) -> PyResult<RetargetSupport> {
    CapabilitiesWire {
        retarget: name.to_owned(),
        request: "on_grid".to_owned(),
        licensed_route: None,
        boundary: "in_process".to_owned(),
        resume: None,
    }
    .to_capabilities()
    .map(|c| c.retarget)
    .map_err(|e| invalid("recalc.invalid_declaration", e.to_string()))
}

fn column_vec(array: &PyReadonlyArray1<'_, f64>) -> Vec<f64> {
    array.as_array().to_vec()
}

/// Validated named columns and the position lookup of a name.
struct Frame {
    names: Vec<String>,
    columns: Columns,
    rows: usize,
}

impl Frame {
    fn new(names: &[String], columns: &[PyReadonlyArray1<'_, f64>]) -> PyResult<Self> {
        if names.is_empty() || names.len() != columns.len() {
            return Err(invalid(
                "recalc.invalid_data",
                "one named numeric column per name is required",
            ));
        }
        for (i, name) in names.iter().enumerate() {
            if name.trim().is_empty() || names[..i].contains(name) {
                return Err(invalid(
                    "recalc.invalid_data",
                    "column names must be unique and non-empty",
                ));
            }
        }
        let columns: Columns = names.iter().cloned().zip(columns.iter().map(column_vec)).collect();
        let rows = columns[0].1.len();
        if rows == 0
            || columns.iter().any(|(_, v)| v.len() != rows || v.iter().any(|x| !x.is_finite()))
        {
            return Err(invalid("recalc.invalid_data", "columns must be finite and equally long"));
        }
        Ok(Self { names: names.to_vec(), columns, rows })
    }

    fn position(&self, name: &str) -> PyResult<u32> {
        let at = self.names.iter().position(|n| n == name).ok_or_else(|| {
            invalid("recalc.unknown_variable", format!("no column named `{name}`"))
        })?;
        u32::try_from(at).map_err(|_| invalid("recalc.invalid_data", "too many columns"))
    }

    fn positions(&self, names: &[String]) -> PyResult<Vec<u32>> {
        names.iter().map(|n| self.position(n)).collect()
    }

    fn edges(&self, edges: &[(String, String)]) -> PyResult<Vec<(u32, u32)>> {
        edges.iter().map(|(a, b)| Ok((self.position(a)?, self.position(b)?))).collect()
    }

    fn target(
        &self,
        weights: Option<&PyReadonlyArray1<'_, f64>>,
        depends_on: &[String],
    ) -> PyResult<Option<TargetWeights>> {
        let Some(weights) = weights else {
            if !depends_on.is_empty() {
                return Err(invalid(
                    "recalc.invalid_request",
                    "target_depends_on needs target weights",
                ));
            }
            return Ok(None);
        };
        let weights = column_vec(weights);
        if weights.len() != self.rows || weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err(invalid(
                "recalc.invalid_request",
                "target weights must be finite, non-negative and one per row",
            ));
        }
        let depends_on =
            self.positions(depends_on)?.into_iter().map(VariableId::from_raw).collect();
        Ok(Some(TargetWeights { weights, depends_on }))
    }
}

fn utility_of(benefit_per_unit: f64, cost: f64) -> PyResult<UtilitySpec> {
    if !benefit_per_unit.is_finite() || !cost.is_finite() {
        return Err(invalid("recalc.invalid_request", "utility must be finite"));
    }
    Ok(UtilitySpec { benefit_per_unit, cost })
}

fn quantity_of(name: &str, arm: Option<u32>) -> PyResult<ScoreQuantity> {
    match (name, arm) {
        ("average_effect", None) => Ok(ScoreQuantity::AverageEffect),
        ("interaction", None) => Ok(ScoreQuantity::Interaction),
        ("cell_minus_control", Some(arm)) => Ok(ScoreQuantity::CellMinusControl { arm }),
        ("cell_minus_control", None) => {
            Err(invalid("recalc.invalid_request", "cell_minus_control needs an arm"))
        }
        ("average_effect" | "interaction", Some(_)) => {
            Err(invalid("recalc.invalid_request", "only cell_minus_control takes an arm"))
        }
        (other, _) => Err(invalid(
            "recalc.invalid_request",
            format!(
                "quantity must be average_effect, interaction or cell_minus_control, got `{other}`"
            ),
        )),
    }
}

fn context(seed: u64, threads: Option<u32>) -> ExecutionContext {
    let mut ctx = ExecutionContext::production(seed, crate::resolve_user_threads(threads));
    // The route proves reuse by counted fits; no persistent cache may stand in for a fit.
    ctx.cache_policy = antecedent_core::CachePolicy::disabled();
    ctx
}

fn counts_wire(counts: &StageCounts) -> CountsWire {
    CountsWire {
        identifications: counts.identifications,
        fold_fits: counts.fold_fits,
        score_computations: counts.score_computations,
        reweights: counts.reweights,
        decisions: counts.decisions,
    }
}

fn result_json(outcome: &RecalcOutcome, receipt: &serde_json::Value) -> String {
    serde_json::json!({
        "plan": plan_to_wire(&outcome.plan),
        "receipt": receipt,
        "law": { "ate": outcome.law.ate, "std_error": outcome.law.std_error },
        "decision": {
            "net_benefit": outcome.decision.net_benefit,
            "treat": outcome.decision.treat,
        },
    })
    .to_string()
}

/// Seal the verified receipt of an in-process run as a `recalc_receipt_v1` artifact.
fn finish_sealed(
    outcome: &RecalcOutcome,
    previous: &StageIdentities,
    current: &StageIdentities,
    capabilities: &RecalcCapabilities,
) -> PyResult<RunPayload> {
    let counts: BTreeMap<Stage, CountsWire> = outcome
        .receipt
        .entries()
        .iter()
        .map(|entry| (entry.stage, counts_wire(&entry.counts)))
        .collect();
    let artifact =
        RecalcReceiptArtifact::seal(previous, current, capabilities, &counts).map_err(py_msg)?;
    if artifact.plan() != &outcome.plan {
        return Err(py_msg("recalc receipt plan differs from the executed plan"));
    }
    let identity = artifact.receipt_identity().to_owned();
    let bytes =
        artifact.to_bytes(&format!("recalc-receipt-{}", &identity[..16])).map_err(py_msg)?;
    let meta = serde_json::to_value(artifact.meta()).map_err(py_msg)?;
    Ok((Some(result_json(outcome, &meta)), Some(bytes), None))
}

fn export_bytes(table: &FrozenScoreTable) -> PyResult<ExportPayload> {
    let identity = table.identity().to_owned();
    let bytes = table.to_bytes(&format!("frozen-scores-{}", &identity[..16])).map_err(py_msg)?;
    Ok((Some((identity, bytes)), None))
}

/// The mean-functional score columns `(arm, scores)` of a table, for independent checks.
fn mean_columns(table: &ScoreTable) -> Vec<(u32, Vec<f64>)> {
    table
        .columns
        .iter()
        .enumerate()
        .filter(|(_, c)| c.threshold.is_none())
        .filter_map(|(at, c)| table.column(at).ok().map(|v| (c.arm, v.to_vec())))
        .collect()
}

// -- cell session -------------------------------------------------------------------------

/// Durable selective-recalculation handle for the cell-AIPW route.
#[pyclass(name = "CellSessionHandle")]
pub struct PyCellSession {
    inner: CellSession,
    retarget: RetargetSupport,
}

impl PyCellSession {
    fn capabilities(&self) -> RecalcCapabilities {
        RecalcCapabilities {
            retarget: self.retarget,
            request: RequestSupport::OnGrid,
            boundary: Boundary::InProcess,
        }
    }
}

fn cell_request(
    names: &[String],
    columns: &[PyReadonlyArray1<'_, f64>],
    edges: &[(String, String)],
    treatments: &[String],
    outcome: &str,
    adjustment: &[String],
    quantity: &str,
    arm: Option<u32>,
    folds: Option<usize>,
    target_weights: Option<&PyReadonlyArray1<'_, f64>>,
    target_depends_on: &[String],
    benefit_per_unit: f64,
    cost: f64,
) -> PyResult<CellRequest> {
    let frame = Frame::new(names, columns)?;
    let treatment_ids = frame.positions(treatments)?;
    if treatment_ids.is_empty() {
        return Err(invalid("recalc.invalid_request", "at least one treatment is required"));
    }
    let mut estimator = CellSaturatedAipw::new();
    if let Some(folds) = folds {
        estimator.folds = folds;
    }
    let spec = CellSpec {
        edges: frame.edges(edges)?,
        treatments: treatment_ids,
        outcome: frame.position(outcome)?,
        adjustment: frame.positions(adjustment)?,
        estimator,
        quantity: quantity_of(quantity, arm)?,
        target: frame.target(target_weights, target_depends_on)?,
        utility: utility_of(benefit_per_unit, cost)?,
    };
    Ok(CellRequest { columns: frame.columns, spec })
}

#[pymethods]
impl PyCellSession {
    /// An empty in-process session.
    #[new]
    #[pyo3(signature = (retarget="licensed"))]
    fn new(retarget: &str) -> PyResult<Self> {
        let retarget = retarget_of(retarget)?;
        let mut inner = CellSession::new();
        inner.set_retarget_support(retarget);
        Ok(Self { inner, retarget })
    }

    /// Declare the estimator's retarget license for later plans.
    fn set_retarget_support(&mut self, retarget: &str) -> PyResult<()> {
        self.retarget = retarget_of(retarget)?;
        self.inner.set_retarget_support(self.retarget);
        Ok(())
    }

    /// Whether the session holds live cell scores.
    fn is_live(&self) -> bool {
        self.inner.is_live()
    }

    /// The identities of the last successful run, as JSON.
    fn identities_json(&self) -> PyResult<String> {
        serde_json::to_string(&identities_to_wire(self.inner.identities())).map_err(py_msg)
    }

    /// The live frozen mean-functional score columns `(arm, scores)`, for independent checks;
    /// `None` when no scores are live.
    fn score_columns(&self) -> Option<Vec<(u32, Vec<f64>)>> {
        self.inner.score_table().map(mean_columns)
    }

    /// The plan the request would run under, without running it: JSON.
    #[pyo3(signature = (names, columns, edges, treatments, outcome, adjustment, quantity,
        benefit_per_unit, cost, *, arm=None, folds=None, target_weights=None,
        target_depends_on=None, seed=1, threads=None))]
    fn plan(
        &self,
        names: Vec<String>,
        columns: Vec<PyReadonlyArray1<'_, f64>>,
        edges: Vec<(String, String)>,
        treatments: Vec<String>,
        outcome: &str,
        adjustment: Vec<String>,
        quantity: &str,
        benefit_per_unit: f64,
        cost: f64,
        arm: Option<u32>,
        folds: Option<usize>,
        target_weights: Option<PyReadonlyArray1<'_, f64>>,
        target_depends_on: Option<Vec<String>>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<String> {
        let request = cell_request(
            &names,
            &columns,
            &edges,
            &treatments,
            outcome,
            &adjustment,
            quantity,
            arm,
            folds,
            target_weights.as_ref(),
            &target_depends_on.unwrap_or_default(),
            benefit_per_unit,
            cost,
        )?;
        let plan = self.inner.plan(&request, &context(seed, threads));
        serde_json::to_string(&plan_to_wire(&plan)).map_err(py_msg)
    }

    /// Plan, run only what the plan says must run, count the work and seal the receipt:
    /// `(result json, receipt artifact bytes, refusal json)`.
    #[pyo3(signature = (names, columns, edges, treatments, outcome, adjustment, quantity,
        benefit_per_unit, cost, *, arm=None, folds=None, target_weights=None,
        target_depends_on=None, seed=1, threads=None))]
    fn execute(
        &mut self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<PyReadonlyArray1<'_, f64>>,
        edges: Vec<(String, String)>,
        treatments: Vec<String>,
        outcome: &str,
        adjustment: Vec<String>,
        quantity: &str,
        benefit_per_unit: f64,
        cost: f64,
        arm: Option<u32>,
        folds: Option<usize>,
        target_weights: Option<PyReadonlyArray1<'_, f64>>,
        target_depends_on: Option<Vec<String>>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<RunPayload> {
        let request = cell_request(
            &names,
            &columns,
            &edges,
            &treatments,
            outcome,
            &adjustment,
            quantity,
            arm,
            folds,
            target_weights.as_ref(),
            &target_depends_on.unwrap_or_default(),
            benefit_per_unit,
            cost,
        )?;
        drop(columns);
        drop(target_weights);
        self.inner.set_retarget_support(self.retarget);
        let ctx = context(seed, threads);
        let capabilities = self.capabilities();
        let previous = self.inner.identities().clone();
        let planned = self.inner.plan(&request, &ctx);
        // Move the session into the detached run and take it back afterwards. A panic or an
        // interrupt loses the live scores, so the next run recomputes every stage.
        let mut session = std::mem::take(&mut self.inner);
        let (session, ran) = detach_catch(py, move || {
            let ran =
                classify(execute_cell_with_receipt(&mut session, &request, &ctx), Some(&planned));
            Ok((session, ran))
        })?;
        self.inner = session;
        match ran {
            Ran::Done(outcome) => {
                finish_sealed(&outcome, &previous, self.inner.identities(), &capabilities)
            }
            Ran::Refused(json) => Ok((None, None, Some(json))),
            Ran::Failed(error) => Err(error),
        }
    }

    /// Export the live cell scores as a portable `frozen_scores_v1` artifact:
    /// `((identity, bytes), refusal json)`.
    fn export_frozen_scores(&self) -> PyResult<ExportPayload> {
        match self.inner.export_scores() {
            Ok(table) => export_bytes(&table),
            Err(error) => export_refusal(error),
        }
    }
}

// -- cross-fit session --------------------------------------------------------------------

/// Durable selective-recalculation handle for the cross-fitted AIPW average effect, so its
/// frozen scores can be exported.
#[pyclass(name = "CrossfitSessionHandle")]
pub struct PyCrossfitSession {
    inner: RecalcSession,
    input_rows: u64,
}

#[pymethods]
impl PyCrossfitSession {
    /// An empty in-process session with a licensed retarget.
    #[new]
    fn new() -> Self {
        let mut inner = RecalcSession::new();
        inner.set_retarget_support(RetargetSupport::Licensed);
        Self { inner, input_rows: 0 }
    }

    /// Whether the session holds a live prepared study.
    fn is_live(&self) -> bool {
        self.inner.is_live()
    }

    /// Plan, run only what the plan says must run, count the work and seal the receipt:
    /// `(result json, receipt artifact bytes, refusal json)`.
    #[pyo3(signature = (names, columns, edges, treatment, outcome, benefit_per_unit, cost, *,
        target_weights=None, target_depends_on=None, seed=1, threads=None))]
    fn execute(
        &mut self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<PyReadonlyArray1<'_, f64>>,
        edges: Vec<(String, String)>,
        treatment: &str,
        outcome: &str,
        benefit_per_unit: f64,
        cost: f64,
        target_weights: Option<PyReadonlyArray1<'_, f64>>,
        target_depends_on: Option<Vec<String>>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<RunPayload> {
        let frame = Frame::new(&names, &columns)?;
        let treatment = frame.position(treatment)?;
        let outcome = frame.position(outcome)?;
        if treatment == outcome {
            return Err(invalid("recalc.invalid_request", "treatment and outcome must differ"));
        }
        let request = RecalcRequest {
            edges: frame.edges(&edges)?,
            treatment,
            outcome,
            estimator: AipwAte::new().with_bootstrap_replicates(0),
            target: frame
                .target(target_weights.as_ref(), &target_depends_on.unwrap_or_default())?,
            utility: utility_of(benefit_per_unit, cost)?,
            columns: frame.columns,
        };
        let rows = u64::try_from(frame.rows).unwrap_or(u64::MAX);
        drop(columns);
        drop(target_weights);
        let ctx = context(seed, threads);
        let capabilities = RecalcCapabilities {
            retarget: RetargetSupport::Licensed,
            request: RequestSupport::OnGrid,
            boundary: Boundary::InProcess,
        };
        let previous = self.inner.identities().clone();
        let planned = self.inner.plan(&request, &ctx);
        let mut session = std::mem::take(&mut self.inner);
        let (session, ran) = detach_catch(py, move || {
            let ran = classify(execute_with_receipt(&mut session, &request, &ctx), Some(&planned));
            Ok((session, ran))
        })?;
        self.inner = session;
        match ran {
            Ran::Done(outcome) => {
                self.input_rows = rows;
                finish_sealed(&outcome, &previous, self.inner.identities(), &capabilities)
            }
            Ran::Refused(json) => Ok((None, None, Some(json))),
            Ran::Failed(error) => Err(error),
        }
    }

    /// Export the live cross-fit scores as a portable `frozen_scores_v1` artifact:
    /// `((identity, bytes), refusal json)`.
    fn export_frozen_scores(&self) -> PyResult<ExportPayload> {
        match freeze_crossfit_scores(&self.inner, self.input_rows, RetargetSupport::Licensed) {
            Ok(table) => export_bytes(&table),
            Err(error) => export_refusal(error),
        }
    }
}

// -- portable resume ----------------------------------------------------------------------

/// A fresh-process session built only from `frozen_scores_v1` bytes. It holds no data and no
/// fitted model, and runs exactly one operation: retarget the frozen scores.
#[pyclass(name = "ScoreResumeHandle")]
pub struct PyScoreResume {
    inner: ScoreResumeSession,
}

fn resume_request(
    n_variables: u32,
    edges: Vec<(u32, u32)>,
    target_weights: Option<&PyReadonlyArray1<'_, f64>>,
    target_depends_on: &[u32],
    target_row_ids: Option<Vec<u32>>,
    quantity: &str,
    arm: Option<u32>,
    benefit_per_unit: f64,
    cost: f64,
    changed_inputs: &[(String, String)],
) -> PyResult<ScoreResumeRequest> {
    let target = match target_weights {
        None => {
            if !target_depends_on.is_empty() {
                return Err(invalid(
                    "recalc.invalid_request",
                    "target_depends_on needs target weights",
                ));
            }
            None
        }
        Some(weights) => {
            let weights = column_vec(weights);
            if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
                return Err(invalid(
                    "recalc.invalid_request",
                    "target weights must be finite and non-negative",
                ));
            }
            Some(TargetWeights {
                weights,
                depends_on: target_depends_on.iter().copied().map(VariableId::from_raw).collect(),
            })
        }
    };
    let changed = changed_inputs
        .iter()
        .map(|(label, own)| {
            let stage = stage_from_label(label).ok_or_else(|| {
                invalid("recalc.invalid_changed_input", format!("unknown stage `{label}`"))
            })?;
            let own = StageIdentity::from_hex(own).ok_or_else(|| {
                invalid("recalc.invalid_changed_input", "an identity is 64 lowercase hex digits")
            })?;
            Ok((stage, own))
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(ScoreResumeRequest {
        n_variables,
        edges,
        target,
        target_row_ids,
        utility: utility_of(benefit_per_unit, cost)?,
        quantity: quantity_of(quantity, arm)?,
        changed_inputs: changed,
    })
}

#[pymethods]
impl PyScoreResume {
    /// Build from artifact bytes, optionally checked against an identity the caller retained
    /// independently: `(handle, refusal json)`. A changed identity is a refusal; corruption and
    /// unsupported versions raise `CausalSerializationError`.
    #[staticmethod]
    #[pyo3(signature = (artifact, expected_identity=None))]
    fn from_bytes(
        py: Python<'_>,
        artifact: Vec<u8>,
        expected_identity: Option<String>,
    ) -> PyResult<(Option<Self>, Option<String>)> {
        let built = detach_catch(py, move || {
            Ok(ScoreResumeSession::resume_from_score_bytes(&artifact, expected_identity.as_deref()))
        })?;
        match built {
            Ok(inner) => Ok((Some(Self { inner }), None)),
            Err(error) => match artifact_refusal(&error) {
                Some(json) => Ok((None, Some(json))),
                None => Err(artifact_error(&error)),
            },
        }
    }

    /// Identity (hex) of the artifact the session was built from.
    fn artifact_identity(&self) -> String {
        self.inner.artifact_identity().to_owned()
    }

    /// Row count of the frozen scores.
    fn n_rows(&self) -> usize {
        self.inner.score_table().n_rows
    }

    /// The original row ids the target weights are indexed by, in row order.
    fn row_ids(&self) -> Vec<u32> {
        self.inner.score_table().row_index.to_vec()
    }

    /// The frozen mean-functional score columns `(arm, scores)`.
    fn score_columns(&self) -> Vec<(u32, Vec<f64>)> {
        mean_columns(self.inner.score_table())
    }

    /// Whether a resumed run has completed.
    fn has_run(&self) -> bool {
        self.inner.has_run()
    }

    /// The identities of the last successful run (the artifact's workflow before any run).
    fn identities_json(&self) -> PyResult<String> {
        serde_json::to_string(&identities_to_wire(self.inner.identities())).map_err(py_msg)
    }

    /// The plan a retarget would run under, or the refusal: `(plan json, refusal json)`.
    #[pyo3(signature = (n_variables, edges, quantity, benefit_per_unit, cost, *, arm=None,
        target_weights=None, target_depends_on=None, target_row_ids=None,
        changed_inputs=None))]
    fn plan(
        &self,
        n_variables: u32,
        edges: Vec<(u32, u32)>,
        quantity: &str,
        benefit_per_unit: f64,
        cost: f64,
        arm: Option<u32>,
        target_weights: Option<PyReadonlyArray1<'_, f64>>,
        target_depends_on: Option<Vec<u32>>,
        target_row_ids: Option<Vec<u32>>,
        changed_inputs: Option<Vec<(String, String)>>,
    ) -> PyResult<ResumePayload> {
        let request = resume_request(
            n_variables,
            edges,
            target_weights.as_ref(),
            &target_depends_on.unwrap_or_default(),
            target_row_ids,
            quantity,
            arm,
            benefit_per_unit,
            cost,
            &changed_inputs.unwrap_or_default(),
        )?;
        match self.inner.plan(&request) {
            Ok(plan) => {
                Ok((Some(serde_json::to_string(&plan_to_wire(&plan)).map_err(py_msg)?), None))
            }
            Err(RecalcRunError::Refused(plan)) => Ok((None, Some(plan_refusal_json(&plan)))),
            Err(RecalcRunError::Request(detail)) => Ok((None, Some(request_refusal_json(detail)))),
            Err(other) => Err(py_msg(other)),
        }
    }

    /// Retarget the frozen scores by row weights and recompute the law and decision, with zero
    /// fits: `(result json, refusal json)`. A request that needs data or a fit is refused with
    /// the plan's `Unavailable` before any work and leaves the session unchanged.
    #[pyo3(signature = (n_variables, edges, quantity, benefit_per_unit, cost, *, arm=None,
        target_weights=None, target_depends_on=None, target_row_ids=None,
        changed_inputs=None))]
    fn execute_retarget(
        &mut self,
        py: Python<'_>,
        n_variables: u32,
        edges: Vec<(u32, u32)>,
        quantity: &str,
        benefit_per_unit: f64,
        cost: f64,
        arm: Option<u32>,
        target_weights: Option<PyReadonlyArray1<'_, f64>>,
        target_depends_on: Option<Vec<u32>>,
        target_row_ids: Option<Vec<u32>>,
        changed_inputs: Option<Vec<(String, String)>>,
    ) -> PyResult<ResumePayload> {
        let request = resume_request(
            n_variables,
            edges,
            target_weights.as_ref(),
            &target_depends_on.unwrap_or_default(),
            target_row_ids,
            quantity,
            arm,
            benefit_per_unit,
            cost,
            &changed_inputs.unwrap_or_default(),
        )?;
        drop(target_weights);
        // The boundary the plan is made under, as `ScoreResumeSession` declares it: a fresh
        // process holding portable scores until a run has completed, in process afterwards.
        let boundary = if self.inner.has_run() {
            Boundary::InProcess
        } else {
            Boundary::FreshProcess(ResumeContext {
                portable_scores: true,
                supplied_data: true,
                ..ResumeContext::default()
            })
        };
        let capabilities = RecalcCapabilities {
            retarget: RetargetSupport::Licensed,
            request: RequestSupport::OnGrid,
            boundary,
        };
        let previous = self.inner.identities().clone();
        let inner = &mut self.inner;
        let ran = detach_catch(py, move || {
            Ok(classify(execute_resumed_retarget(inner, &request), None))
        })?;
        match ran {
            Ran::Done(outcome) => {
                let meta = record_meta(&outcome, &previous, self.inner.identities(), &capabilities);
                Ok((Some(result_json(&outcome, &meta)), None))
            }
            Ran::Refused(json) => Ok((None, Some(json))),
            Ran::Failed(error) => Err(error),
        }
    }
}

/// The receipt of a resumed run as a record. A fresh-process receipt that reuses the portable
/// score artifact cannot be sealed as `recalc_receipt_v1` (that format refuses a derived stage
/// reused in a fresh process), so the record carries the verified table and counts without an
/// artifact.
fn record_meta(
    outcome: &RecalcOutcome,
    previous: &StageIdentities,
    current: &StageIdentities,
    capabilities: &RecalcCapabilities,
) -> serde_json::Value {
    let plan = plan_to_wire(&outcome.plan);
    let entries: Vec<ReceiptEntryWire> = plan
        .entries
        .iter()
        .map(|row| {
            let counts = stage_from_label(&row.stage)
                .and_then(|stage| outcome.receipt.entry(stage))
                .map(|entry| counts_wire(&entry.counts))
                .unwrap_or_default();
            ReceiptEntryWire {
                stage: row.stage.clone(),
                tag: row.tag.clone(),
                detail: row.detail.clone(),
                status: row.status.clone(),
                identity: row.identity.clone(),
                counts,
            }
        })
        .collect();
    serde_json::json!({
        "version": 1,
        "feature": "resume_record",
        "capabilities": CapabilitiesWire::from_capabilities(capabilities),
        "previous": identities_to_wire(previous),
        "requested": identities_to_wire(current),
        "plan_identity": plan.identity,
        "receipt_identity": outcome.receipt.identity().to_hex(),
        "entries": entries,
        "totals": counts_wire(outcome.receipt.totals()),
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCellSession>()?;
    m.add_class::<PyCrossfitSession>()?;
    m.add_class::<PyScoreResume>()
}
