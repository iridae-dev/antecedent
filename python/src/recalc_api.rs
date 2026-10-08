//! Bounded Python bridge for selective recalculation with a visible receipt (2.3 C2).
//!
//! Rust owns the stage model, the planner, the instrumented execution of the cross-fitted
//! AIPW route, the receipt counts, the portable receipt artifact and every refusal. Python
//! declares a workflow (as per-stage own-input digests for planning, or as raw columns for a
//! run) and reads back JSON. A refusal comes back as structured JSON for the Python layer
//! to raise as its own exception type; corruption and unknown versions raise
//! `CausalSerializationError`.
#![allow(clippy::too_many_arguments)]

use std::collections::BTreeMap;

use antecedent::analysis::recalc_receipt::{
    RecalcOutcome, RecalcRequest, RecalcRunError, RecalcSession, TargetWeights, UtilitySpec,
    execute_with_receipt,
};
use antecedent_core::recalc::{
    Boundary, RecalcCapabilities, RecalcPlan, RefusalReason, ResumeContext, RetargetSupport, Stage,
    StageIdentity,
};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_estimate::AipwAte;
use antecedent_io::recalc_receipt_artifact::{
    CapabilitiesWire, CountsWire, DeclaredStageWire, RecalcReceiptArtifact,
    RecalcReceiptArtifactError, ResumeWire, identities_from_wire, identities_to_wire,
    plan_from_wire, plan_to_wire,
};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{CausalSerializationError, detach_catch, py_err, py_msg, value_err, with_reason_code};

/// Largest declaration JSON accepted.
const MAX_JSON_BYTES: usize = 1024 * 1024;
type RunPayload = (Option<String>, Option<Vec<u8>>, Option<String>);
type ConsumePayload = (Option<String>, Option<String>);

fn invalid(detail: &str, text: impl AsRef<str>) -> PyErr {
    with_reason_code(
        value_err(format!("{detail}: {}", text.as_ref())),
        antecedent_core::reason_code!("invalid_argument"),
    )
}

fn artifact_invalid(error: &RecalcReceiptArtifactError) -> PyErr {
    invalid("recalc.invalid_declaration", error.to_string())
}

fn parse_json<T: serde::de::DeserializeOwned>(text: &str, what: &str) -> PyResult<T> {
    if text.len() > MAX_JSON_BYTES {
        return Err(PyValueError::new_err(format!("recalc {what} declaration is too large")));
    }
    serde_json::from_str(text)
        .map_err(|e| invalid("recalc.invalid_declaration", format!("{what}: {e}")))
}

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
            antecedent_core::recalc::MissingDependency::Fit => {
                "supply a portable fit or scores, or the data snapshot to refit".to_owned()
            }
            antecedent_core::recalc::MissingDependency::Data => {
                "supply a compatible data snapshot".to_owned()
            }
            antecedent_core::recalc::MissingDependency::Provider => {
                "supply a compatible provider callback".to_owned()
            }
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

// -- pure planning ------------------------------------------------------------------------

/// Plan `requested` against `previous` under `capabilities` from declared own-input digests.
///
/// The three arguments are JSON: two lists of `{"stage", "own"}` and one capabilities object.
/// Nothing is run: this decides and explains.
#[pyfunction]
fn plan_recalculation(
    previous_json: &str,
    requested_json: &str,
    capabilities_json: &str,
) -> PyResult<String> {
    let previous: Vec<DeclaredStageWire> = parse_json(previous_json, "previous")?;
    let requested: Vec<DeclaredStageWire> = parse_json(requested_json, "requested")?;
    let capabilities: CapabilitiesWire = parse_json(capabilities_json, "capabilities")?;
    let plan =
        plan_from_wire(&previous, &requested, &capabilities).map_err(|e| artifact_invalid(&e))?;
    serde_json::to_string(&plan_to_wire(&plan)).map_err(py_msg)
}

/// The own-input digest (64 lowercase hex characters) of a stage declaration: `label` and
/// length-prefixed `parts`.
#[pyfunction]
fn recalc_stage_identity(label: &str, parts: Vec<Vec<u8>>) -> String {
    let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
    StageIdentity::of(label, &refs).to_hex()
}

fn report_json(artifact: &RecalcReceiptArtifact) -> PyResult<String> {
    serde_json::to_string(&serde_json::json!({
        "receipt": artifact.meta(),
        "plan": plan_to_wire(artifact.plan()),
    }))
    .map_err(py_msg)
}

fn consume_refusal(error: &RecalcReceiptArtifactError) -> Option<String> {
    let (code, detail, text) = error.refusal()?;
    Some(refusal_json(code, error.stage().unwrap_or(""), detail, error.stage(), None, &text, None))
}

/// Consume a receipt artifact by recomputation: `(report json, refusal json)`.
///
/// `expected_identity` is a receipt identity the caller retained independently; a
/// consistently resealed receipt of another run is refused against it.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity=None))]
fn consume_recalc_receipt(
    py: Python<'_>,
    artifact: Vec<u8>,
    expected_identity: Option<String>,
) -> PyResult<ConsumePayload> {
    detach_catch(py, move || {
        match RecalcReceiptArtifact::from_bytes(&artifact, expected_identity.as_deref()) {
            Ok(done) => Ok((Some(report_json(&done)?), None)),
            Err(error) => match consume_refusal(&error) {
                Some(json) => Ok((None, Some(json))),
                None => Err(match error {
                    RecalcReceiptArtifactError::LimitsExceeded(what) => {
                        invalid("recalc.limits_exceeded", what)
                    }
                    other => CausalSerializationError::new_err(other.to_string()),
                }),
            },
        }
    })
}

// -- session ------------------------------------------------------------------------------

fn capabilities_wire(
    retarget: &str,
    request: &str,
    licensed_route: Option<&str>,
) -> CapabilitiesWire {
    CapabilitiesWire {
        retarget: retarget.to_owned(),
        request: request.to_owned(),
        licensed_route: licensed_route.map(str::to_owned),
        boundary: "in_process".to_owned(),
        resume: None,
    }
}

fn retarget_of(name: &str) -> PyResult<RetargetSupport> {
    capabilities_wire(name, "on_grid", None)
        .to_capabilities()
        .map(|c| c.retarget)
        .map_err(|e| artifact_invalid(&e))
}

fn request_support_of(
    request: &str,
    licensed_route: Option<&str>,
) -> PyResult<antecedent_core::recalc::RequestSupport> {
    capabilities_wire("licensed", request, licensed_route)
        .to_capabilities()
        .map(|c| c.request)
        .map_err(|e| artifact_invalid(&e))
}

fn column_vec(array: &PyReadonlyArray1<'_, f64>) -> Vec<f64> {
    array.as_array().to_vec()
}

/// Validate and assemble one cross-fitted AIPW average-effect request.
fn build_request(
    names: &[String],
    columns: &[PyReadonlyArray1<'_, f64>],
    edges: &[(String, String)],
    treatment: &str,
    outcome: &str,
    target_weights: Option<&PyReadonlyArray1<'_, f64>>,
    target_depends_on: &[String],
    benefit_per_unit: f64,
    cost: f64,
) -> PyResult<RecalcRequest> {
    if names.is_empty() || names.len() != columns.len() {
        return Err(invalid(
            "recalc.invalid_data",
            "one named numeric column per name is required",
        ));
    }
    let position = |name: &str| -> PyResult<u32> {
        let at = names.iter().position(|n| n == name).ok_or_else(|| {
            invalid("recalc.unknown_variable", format!("no column named `{name}`"))
        })?;
        u32::try_from(at).map_err(|_| invalid("recalc.invalid_data", "too many columns"))
    };
    for (i, name) in names.iter().enumerate() {
        if name.trim().is_empty() || names[..i].contains(name) {
            return Err(invalid(
                "recalc.invalid_data",
                "column names must be unique and non-empty",
            ));
        }
    }
    let columns: Vec<(String, Vec<f64>)> =
        names.iter().cloned().zip(columns.iter().map(column_vec)).collect();
    let rows = columns[0].1.len();
    if rows == 0 || columns.iter().any(|(_, v)| v.len() != rows || v.iter().any(|x| !x.is_finite()))
    {
        return Err(invalid("recalc.invalid_data", "columns must be finite and equally long"));
    }
    let treatment = position(treatment)?;
    let outcome = position(outcome)?;
    if treatment == outcome {
        return Err(invalid("recalc.invalid_request", "treatment and outcome must differ"));
    }
    let edges = edges
        .iter()
        .map(|(from, to)| Ok((position(from)?, position(to)?)))
        .collect::<PyResult<Vec<_>>>()?;
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
            if weights.len() != rows || weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
                return Err(invalid(
                    "recalc.invalid_request",
                    "target weights must be finite, non-negative and one per row",
                ));
            }
            let depends_on = target_depends_on
                .iter()
                .map(|name| position(name).map(VariableId::from_raw))
                .collect::<PyResult<Vec<_>>>()?;
            Some(TargetWeights { weights, depends_on })
        }
    };
    if !benefit_per_unit.is_finite() || !cost.is_finite() {
        return Err(invalid("recalc.invalid_request", "utility must be finite"));
    }
    Ok(RecalcRequest {
        columns,
        edges,
        treatment,
        outcome,
        estimator: AipwAte::new().with_bootstrap_replicates(0),
        target,
        utility: UtilitySpec { benefit_per_unit, cost },
    })
}

fn context(seed: u64, threads: Option<u32>) -> ExecutionContext {
    let mut ctx = ExecutionContext::production(seed, crate::resolve_user_threads(threads));
    // The route proves reuse by counted fits; no persistent cache may stand in for a fit.
    ctx.cache_policy = antecedent_core::CachePolicy::disabled();
    ctx
}

fn counts_wire(counts: &antecedent::analysis::recalc_receipt::StageCounts) -> CountsWire {
    CountsWire {
        identifications: counts.identifications,
        fold_fits: counts.fold_fits,
        score_computations: counts.score_computations,
        reweights: counts.reweights,
        decisions: counts.decisions,
    }
}

/// What one detached run produced.
enum Ran {
    Done(Box<RecalcOutcome>),
    Refused(Box<RecalcPlan>),
    NoLiveState(Stage),
    Failed(PyErr),
}

/// Durable selective-recalculation handle for the cross-fitted AIPW average effect.
#[pyclass(name = "RecalcSessionHandle")]
pub struct PyRecalcSession {
    inner: RecalcSession,
    retarget: RetargetSupport,
    request: antecedent_core::recalc::RequestSupport,
    resume: Option<ResumeContext>,
}

impl PyRecalcSession {
    fn capabilities(&self) -> RecalcCapabilities {
        let boundary = match self.resume {
            Some(resume) if !self.inner.is_live() => Boundary::FreshProcess(resume),
            _ => Boundary::InProcess,
        };
        RecalcCapabilities { retarget: self.retarget, request: self.request, boundary }
    }

    fn declare(&mut self) {
        self.inner.set_retarget_support(self.retarget);
        self.inner.set_request_support(self.request);
    }
}

#[pymethods]
impl PyRecalcSession {
    /// An empty in-process session.
    #[new]
    #[pyo3(signature = (retarget="licensed"))]
    fn new(retarget: &str) -> PyResult<Self> {
        let retarget = retarget_of(retarget)?;
        let mut inner = RecalcSession::new();
        inner.set_retarget_support(retarget);
        Ok(Self {
            inner,
            retarget,
            request: antecedent_core::recalc::RequestSupport::OnGrid,
            resume: None,
        })
    }

    /// A fresh-process session that knows only the previous identities (a JSON list of
    /// `{"stage", "own"}`) and what the resume context (a JSON object) names.
    #[staticmethod]
    #[pyo3(signature = (previous_json, resume_json, retarget="licensed"))]
    fn resume(previous_json: &str, resume_json: &str, retarget: &str) -> PyResult<Self> {
        let previous: Vec<DeclaredStageWire> = parse_json(previous_json, "previous")?;
        let resume: ResumeWire = parse_json(resume_json, "resume")?;
        let previous = identities_from_wire(&previous).map_err(|e| artifact_invalid(&e))?;
        let resume = ResumeContext {
            portable_fit: resume.portable_fit,
            portable_scores: resume.portable_scores,
            scores_snapshot_bound: resume.scores_snapshot_bound,
            supplied_data: resume.supplied_data,
            supplied_provider: resume.supplied_provider,
        };
        let retarget = retarget_of(retarget)?;
        let mut inner = RecalcSession::resume(previous, resume);
        inner.set_retarget_support(retarget);
        Ok(Self {
            inner,
            retarget,
            request: antecedent_core::recalc::RequestSupport::OnGrid,
            resume: Some(resume),
        })
    }

    /// Declare the estimator's retarget license for later plans.
    fn set_retarget_support(&mut self, retarget: &str) -> PyResult<()> {
        self.retarget = retarget_of(retarget)?;
        self.declare();
        Ok(())
    }

    /// Declare whether the next request is `on_grid`, `off_grid` or `unsupported`, with the
    /// separately licensed route that serves it, when one exists.
    #[pyo3(signature = (request, licensed_route=None))]
    fn set_request_support(&mut self, request: &str, licensed_route: Option<&str>) -> PyResult<()> {
        self.request = request_support_of(request, licensed_route)?;
        self.declare();
        Ok(())
    }

    /// Whether the session holds a live prepared study.
    fn is_live(&self) -> bool {
        self.inner.is_live()
    }

    /// The identities of the last successful run (or the resumed ones), as JSON.
    fn identities_json(&self) -> PyResult<String> {
        serde_json::to_string(&identities_to_wire(self.inner.identities())).map_err(py_msg)
    }

    /// The capabilities the next plan is made under, as JSON.
    fn capabilities_json(&self) -> PyResult<String> {
        serde_json::to_string(&CapabilitiesWire::from_capabilities(&self.capabilities()))
            .map_err(py_msg)
    }

    /// The contrast scores `phi_1 - phi_0` of the live frozen score table, for independent
    /// checks; `None` when no study is live.
    fn score_contrast(&self) -> Option<Vec<f64>> {
        let table = self.inner.score_table()?;
        let column = |arm: u32| {
            let at = table.columns.iter().position(|c| c.arm == arm && c.threshold.is_none())?;
            table.column(at).ok().map(<[f64]>::to_vec)
        };
        let (control, active) = (column(0)?, column(1)?);
        Some(active.iter().zip(&control).map(|(a, c)| a - c).collect())
    }

    /// The plan `request` would run under, without running it: JSON.
    #[pyo3(signature = (names, columns, edges, treatment, outcome, benefit_per_unit, cost, *,
        target_weights=None, target_depends_on=None, seed=1, threads=None))]
    fn plan(
        &self,
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
    ) -> PyResult<String> {
        let request = build_request(
            &names,
            &columns,
            &edges,
            treatment,
            outcome,
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
        let request = build_request(
            &names,
            &columns,
            &edges,
            treatment,
            outcome,
            target_weights.as_ref(),
            &target_depends_on.unwrap_or_default(),
            benefit_per_unit,
            cost,
        )?;
        drop(columns);
        drop(target_weights);
        self.declare();
        let ctx = context(seed, threads);
        let capabilities = self.capabilities();
        let previous = self.inner.identities().clone();
        let planned = self.inner.plan(&request, &ctx);
        // Move the session into the detached run and take it back afterwards. A panic or an
        // interrupt loses the live study, so the next run recomputes every stage.
        let mut session = std::mem::take(&mut self.inner);
        let (session, ran) = detach_catch(py, move || {
            let ran = match execute_with_receipt(&mut session, &request, &ctx) {
                Ok(outcome) => Ran::Done(Box::new(outcome)),
                Err(RecalcRunError::Refused(plan)) => Ran::Refused(plan),
                Err(RecalcRunError::NoLiveState(stage)) => Ran::NoLiveState(stage),
                Err(RecalcRunError::Request(detail)) => {
                    Ran::Failed(invalid(detail, "the request is malformed"))
                }
                Err(RecalcRunError::Execution(error)) => Ran::Failed(py_err(error)),
                Err(RecalcRunError::Receipt(error)) => Ran::Failed(py_msg(error)),
            };
            Ok((session, ran))
        })?;
        self.inner = session;
        match ran {
            Ran::Done(outcome) => self.finish(&outcome, &previous, &capabilities),
            Ran::Refused(plan) => Ok((None, None, Some(plan_refusal_json(&plan)))),
            Ran::NoLiveState(stage) => Ok((
                None,
                None,
                Some(refusal_json(
                    antecedent_core::reason_code!("score_table_unavailable"),
                    &stage.label(),
                    "recalc.no_live_state",
                    Some(&stage.label()),
                    Some(
                        "run the stage in this process, or resume with supplied data so it is \
                         recomputed instead of reused"
                            .to_owned(),
                    ),
                    &format!("no live artifact to reuse for stage {stage}"),
                    Some(&planned),
                )),
            )),
            Ran::Failed(error) => Err(error),
        }
    }
}

impl PyRecalcSession {
    fn finish(
        &self,
        outcome: &RecalcOutcome,
        previous: &antecedent_core::recalc::StageIdentities,
        capabilities: &RecalcCapabilities,
    ) -> PyResult<RunPayload> {
        let counts: BTreeMap<Stage, CountsWire> = outcome
            .receipt
            .entries()
            .iter()
            .map(|entry| (entry.stage, counts_wire(&entry.counts)))
            .collect();
        let artifact =
            RecalcReceiptArtifact::seal(previous, self.inner.identities(), capabilities, &counts)
                .map_err(py_msg)?;
        if artifact.plan() != &outcome.plan {
            return Err(py_msg("recalc receipt plan differs from the executed plan"));
        }
        let identity = artifact.receipt_identity().to_owned();
        let bytes =
            artifact.to_bytes(&format!("recalc-receipt-{}", &identity[..16])).map_err(py_msg)?;
        let result = serde_json::json!({
            "plan": plan_to_wire(&outcome.plan),
            "receipt": artifact.meta(),
            "law": { "ate": outcome.law.ate, "std_error": outcome.law.std_error },
            "decision": {
                "net_benefit": outcome.decision.net_benefit,
                "treat": outcome.decision.treat,
            },
        });
        Ok((Some(result.to_string()), Some(bytes), None))
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(plan_recalculation, m)?)?;
    m.add_function(wrap_pyfunction!(recalc_stage_identity, m)?)?;
    m.add_function(wrap_pyfunction!(consume_recalc_receipt, m)?)?;
    m.add_class::<PyRecalcSession>()
}
