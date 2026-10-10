//! Shared-plan recalculation of checked IV, sharp RD and linear front-door effects.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::PreparedStudy;
use super::recalc_receipt::{
    Counter, DecisionValue, LawValue, RecalcOutcome, RecalcRunError, ReceiptRecorder, UtilitySpec,
    decide,
};
use crate::graph::{Dag, DenseNodeId};
use crate::{CausalError, IdentifierId, RefuteSuite, Study, StudyResult};
use antecedent_core::recalc::{
    Boundary, RecalcCapabilities, RecalcPlan, RequestSupport, ResumeContext, RetargetSupport,
    Stage, StageIdentities, StageIdentity, StageStatus,
};
use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_estimate::{FrontDoorTwoStage, TwoStageLeastSquares};
use antecedent_stats::fit_counts::count_least_squares_solves;
use antecedent_stats::twosls::FirstStageDiagnostics;
use std::sync::Arc;

/// Declared design, causal roles and numerical window. These affect identification and fit.
#[derive(Clone, Debug)]
pub enum DesignModel {
    /// A single binary excluded instrument and two-stage least squares.
    /// Actual Anderson–Rubin diagnostics are the uncertainty product, never a Wald SE.
    /// A decision requires a finite connected AR set; unavailable/unbounded sets refuse.
    Iv2Sls {
        /// Instrument coordinate.
        instrument: u32,
    },
    /// Checked sharp discontinuity and one local linear interaction regression.
    Rd {
        /// Running variable coordinate.
        running_variable: u32,
        /// Treatment assignment threshold.
        cutoff: f64,
        /// Symmetric local half-width.
        bandwidth: f64,
    },
    /// Checked linear path product through one observed mediator.
    Frontdoor {
        /// Mediator coordinate.
        mediator: u32,
    },
}
/// Input frame, causal graph, effect roles and utility of one supported design.
#[derive(Clone, Debug)]
pub struct DesignRequest {
    /// Columns in stable variable-id order; missing values use complete-case semantics.
    pub columns: Vec<(String, Vec<f64>)>,
    /// Directed edges over those stable coordinates.
    pub edges: Vec<(u32, u32)>,
    /// Treatment coordinate, compared at zero and one.
    pub treatment: u32,
    /// Outcome coordinate.
    pub outcome: u32,
    /// Design declaration.
    pub model: DesignModel,
    /// Utility rule. A change to this alone reuses the checked result.
    pub utility: UtilitySpec,
}
fn literal(label: &str, value: &str) -> StageIdentity {
    StageIdentity::of(label, &[value.as_bytes()])
}
impl DesignRequest {
    /// The declared inputs of the shared causal and numerical stage graph.
    #[must_use]
    pub fn identities(&self, ctx: &ExecutionContext) -> StageIdentities {
        let mut ids = StageIdentities::new();
        let mut edges = self.edges.clone();
        edges.sort_unstable();
        edges.dedup();
        ids.set(
            Stage::Graph,
            literal("design.graph", &format!("{}:{edges:?}", self.columns.len())),
        );
        ids.set(
            Stage::Query,
            literal(
                "design.query",
                &format!("{}:{}:{:?}", self.treatment, self.outcome, self.model),
            ),
        );
        ids.set(Stage::Regime, literal("design.regime", "static.0_1"));
        ids.set(Stage::Evidence, literal("design.evidence", "checked.observational.design"));
        ids.set(Stage::SourcePopulation, literal("design.source", "observed.design_population"));
        ids.set(Stage::TargetPopulation, literal("design.target", "fixed.design_population"));
        let mut data = Vec::new();
        for (name, values) in &self.columns {
            data.push(name.as_bytes().to_vec());
            data.push(values.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect());
        }
        ids.set(
            Stage::DataSnapshot,
            StageIdentity::of(
                "design.snapshot",
                &data.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            ),
        );
        ids.set(
            Stage::RowDesign,
            literal(
                "design.rows",
                &format!("complete_case.{}", self.columns.first().map_or(0, |(_, v)| v.len())),
            ),
        );
        ids.set(Stage::TreatmentGrid, literal("design.grid", &format!("{:?}", self.model)));
        ids.set(
            Stage::LearnerFoldsRng,
            literal(
                "design.estimator",
                &format!("point.analytic:{:?}:{}", self.model, ctx.rng.master_seed()),
            ),
        );
        ids.set(
            Stage::Utility,
            StageIdentity::of(
                "design.utility",
                &[
                    &self.utility.benefit_per_unit.to_bits().to_le_bytes(),
                    &self.utility.cost.to_bits().to_le_bytes(),
                ],
            ),
        );
        ids.set(Stage::Identification, literal("design.identification", "checked.design.program"));
        ids.set(
            Stage::ScoreArtifact,
            literal("design.fit", "checked.engine.point_and_diagnostics"),
        );
        ids.set(Stage::Law, literal("design.law", "fixed.design_contrast"));
        ids.set(Stage::Decision, literal("design.decision", "net_benefit"));
        ids
    }
    fn validate(&self) -> Result<(), RecalcRunError> {
        self.utility.validate()?;
        let n = self.columns.first().map_or(0, |(_, v)| v.len());
        let names: std::collections::BTreeSet<_> =
            self.columns.iter().map(|(name, _)| name).collect();
        if n == 0
            || names.len() != self.columns.len()
            || self.columns.iter().any(|(name, v)| {
                name.is_empty() || v.len() != n || v.iter().any(|x| x.is_infinite())
            })
        {
            return Err(RecalcRunError::Request("recalc.invalid_data"));
        }
        let design_role = match self.model {
            DesignModel::Iv2Sls { instrument } => instrument,
            DesignModel::Rd { running_variable, .. } => running_variable,
            DesignModel::Frontdoor { mediator } => mediator,
        };
        let roles = [self.treatment, self.outcome, design_role];
        if roles.iter().any(|&id| id as usize >= self.columns.len())
            || roles.into_iter().collect::<std::collections::BTreeSet<_>>().len() != 3
        {
            return Err(RecalcRunError::Request("recalc.invalid_design_roles"));
        }
        if let DesignModel::Rd { cutoff, bandwidth, .. } = self.model {
            if !cutoff.is_finite() || !bandwidth.is_finite() || bandwidth <= 0. {
                return Err(RecalcRunError::Request("recalc.invalid_rd_window"));
            }
        }
        Ok(())
    }
    fn data(&self) -> Result<TabularData, RecalcRunError> {
        TabularData::from_f64_columns(
            self.columns.iter().map(|(n, v)| (n.as_str(), v.as_slice())).collect::<Vec<_>>(),
        )
        .map_err(|_| RecalcRunError::Request("recalc.invalid_data"))
    }
    fn study(&self, data: TabularData) -> Result<Study, RecalcRunError> {
        let mut graph = Dag::with_variables(
            u32::try_from(self.columns.len())
                .map_err(|_| RecalcRunError::Request("recalc.invalid_graph"))?,
        );
        let mut edges = self.edges.clone();
        edges.sort_unstable();
        edges.dedup();
        for (a, b) in edges {
            graph
                .insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b))
                .map_err(|_| RecalcRunError::Request("recalc.invalid_graph"))?;
        }
        let builder = Study::tabular(data)
            .graph(graph)
            .query(AverageEffectQuery::with_levels(
                VariableId::from_raw(self.treatment),
                VariableId::from_raw(self.outcome),
                0.,
                1.,
            ))
            .refute(RefuteSuite::None);
        let builder = match self.model {
            DesignModel::Iv2Sls { .. } => builder
                .identifier(IdentifierId::Iv)
                .estimator(TwoStageLeastSquares::new().with_bootstrap_replicates(0)),
            DesignModel::Rd { running_variable, cutoff, bandwidth } => builder
                .identifier(IdentifierId::RdSharp)
                .estimator(crate::EstimatorId::RdSharp)
                .bootstrap_replicates(0)
                .rd_config(VariableId::from_raw(running_variable), cutoff, bandwidth),
            DesignModel::Frontdoor { .. } => builder
                .identifier(IdentifierId::Frontdoor)
                .estimator(FrontDoorTwoStage::new().with_bootstrap_replicates(0)),
        };
        Ok(builder.build()?)
    }
    fn check_design(&self, prepared: &PreparedStudy) -> Result<(), RecalcRunError> {
        let matches = match self.model {
            DesignModel::Iv2Sls { instrument } => prepared.checked_iv().is_some_and(|p| {
                p.lowering().instruments.as_ref() == [VariableId::from_raw(instrument)]
            }),
            DesignModel::Rd { running_variable, cutoff, bandwidth } => {
                prepared.checked_rd_preparation().is_some_and(|p| {
                    p.lowering().running_variable.raw() == running_variable
                        && p.lowering().cutoff.to_bits() == cutoff.to_bits()
                        && p.lowering().bandwidth.to_bits() == bandwidth.to_bits()
                })
            }
            DesignModel::Frontdoor { mediator } => {
                prepared.checked_frontdoor_linear().is_some_and(|p| {
                    p.lowering().mediators.as_ref() == [VariableId::from_raw(mediator)]
                })
            }
        };
        if matches { Ok(()) } else { Err(RecalcRunError::Request("recalc.design_not_checked")) }
    }
}
/// A typed execution refusal; IV weakness preserves the existing engine's actual AR output.
#[derive(Debug)]
pub enum DesignRunError {
    /// Shared planner/request or existing engine refusal.
    Recalc(RecalcRunError),
    /// A finite decision bound is unavailable. The fitted point and AR diagnostics survive.
    WeakInstrument {
        /// Actual IV point, available for inspection without a supported decision.
        point: f64,
        /// Actual successful numerical solves performed before this refusal.
        model_fits: u64,
        /// Actual fitted diagnostics, including AR set/withheld reason.
        diagnostics: Box<FirstStageDiagnostics>,
    },
}
impl From<RecalcRunError> for DesignRunError {
    fn from(e: RecalcRunError) -> Self {
        Self::Recalc(e)
    }
}
impl From<CausalError> for DesignRunError {
    fn from(e: CausalError) -> Self {
        Self::Recalc(e.into())
    }
}
impl std::fmt::Display for DesignRunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Recalc(e) => write!(f, "{e}"),
            Self::WeakInstrument { diagnostics, .. } => write!(
                f,
                "weak_instruments: recalc.iv_decision_unavailable; actual Anderson–Rubin diagnostics: {diagnostics:?}"
            ),
        }
    }
}
impl std::error::Error for DesignRunError {}
struct Live {
    prepared: Arc<PreparedStudy>,
    result: Arc<StudyResult>,
    ctx: ExecutionContext,
    law: LawValue,
    decision: DecisionValue,
}
/// Actual immutable checked result retained for summary/utility reuse.
pub struct DesignSession {
    previous: StageIdentities,
    live: Option<Live>,
    boundary: Boundary,
}
impl Default for DesignSession {
    fn default() -> Self {
        Self::new()
    }
}
impl DesignSession {
    /// Empty in-process session.
    #[must_use]
    pub fn new() -> Self {
        Self { previous: StageIdentities::new(), live: None, boundary: Boundary::InProcess }
    }
    /// Identities do not supply a fit. Only supplied raw data can rebuild a fresh session.
    #[must_use]
    pub fn resume(previous: StageIdentities, resume: ResumeContext) -> Self {
        Self {
            previous,
            live: None,
            boundary: Boundary::FreshProcess(ResumeContext {
                supplied_data: resume.supplied_data,
                ..ResumeContext::default()
            }),
        }
    }
    /// Last successful stage inputs.
    #[must_use]
    pub const fn identities(&self) -> &StageIdentities {
        &self.previous
    }
    /// Whether this session actually retains the checked result.
    #[must_use]
    pub const fn is_live(&self) -> bool {
        self.live.is_some()
    }
    /// Actual process boundary.
    #[must_use]
    pub const fn boundary(&self) -> Boundary {
        self.boundary
    }
    /// Borrow the actual typed estimator output, including IV AR diagnostics.
    #[must_use]
    pub fn result(&self) -> Option<&StudyResult> {
        self.live.as_ref().map(|l| l.result.as_ref())
    }
    /// Plan against actual retained state. Population reweighting is not licensed.
    #[must_use]
    pub fn plan(&self, r: &DesignRequest, ctx: &ExecutionContext) -> RecalcPlan {
        RecalcPlan::plan(
            &self.previous,
            &r.identities(ctx),
            &RecalcCapabilities {
                retarget: RetargetSupport::NotDeclared,
                request: RequestSupport::OnGrid,
                boundary: self.boundary,
            },
        )
    }
    /// Export through the existing result/program artifact consumer.
    /// RD artifacts explicitly retain its unresolved checked-operation dependency;
    /// fresh sessions require supplied raw data to execute that operation again.
    /// # Errors
    /// No live checked result, or the existing artifact consumer's schema/identity errors.
    pub fn export_result(&self) -> Result<Vec<u8>, DesignRunError> {
        let live = self.live.as_ref().ok_or(RecalcRunError::NoLiveState(Stage::ScoreArtifact))?;
        Ok(live.prepared.encode_contracted_result(&live.result, "recalc-design", &live.ctx)?)
    }
}
fn recomputed(plan: &RecalcPlan, stage: Stage) -> bool {
    matches!(plan.status(stage), Some(StageStatus::Recomputed { .. }))
}
fn fit_guard(r: &DesignRequest, ctx: &ExecutionContext) -> Result<(), RecalcRunError> {
    let n = r.columns.first().map_or(0, |(_, v)| v.len()) as u64;
    let p = r.columns.len() as u64;
    let bytes = n
        .saturating_mul(p)
        .saturating_mul(128)
        .saturating_add(p.saturating_mul(p).saturating_mul(128));
    if ctx
        .memory
        .hard_limit_bytes
        .into_iter()
        .chain(ctx.memory.soft_limit_bytes)
        .any(|limit| bytes > limit)
    {
        return Err(RecalcRunError::Request("recalc.memory_budget_exceeded"));
    }
    Ok(())
}
fn summarize_checked_result(
    result: &StudyResult,
    recorder: &mut ReceiptRecorder,
) -> Result<LawValue, RecalcRunError> {
    antecedent_core::execution_attempt::run_operation(
        antecedent_core::execution_attempt::Operation::LawSummary,
        || {
            let effect = result
                .estimate
                .as_effect()
                .ok_or(RecalcRunError::Request("recalc.design_effect_unavailable"))?;
            if !effect.ate.is_finite() {
                return Err(RecalcRunError::Request("recalc.design_point_unavailable"));
            }
            let law = LawValue { ate: effect.ate, std_error: effect.se_analytic };
            recorder.record(Counter::LawSummary, 1);
            Ok(law)
        },
    )
}

/// A newly executed, source-bound consumer result. Counts describe this execution,
/// rather than the historical producer's work. The session retains its actual fit.
pub struct DesignReplay {
    /// Fresh checked state supplied by actual raw data and graph execution.
    pub session: DesignSession,
    /// Fresh shared-plan execution and its measured work receipt.
    pub outcome: RecalcOutcome,
    /// Identity of the independently decoded historical artifact.
    pub artifact_digest: [u8; 32],
}

/// Independently consume a design artifact and replay it against supplied raw data.
///
/// The existing reader remains unchanged. Its missing RD operation is discharged only
/// by running the checked engine and matching the complete data, program, inference,
/// execution and scientific-result bindings. The producing seed must be supplied.
/// # Errors
/// Missing raw data, malformed or unsupported artifact, checked-engine refusal, or any
/// mismatch of the supplied data/configuration/execution context and original result.
pub fn consume_design_with_data(
    bytes: &[u8],
    request: Option<&DesignRequest>,
    ctx: &ExecutionContext,
) -> Result<DesignReplay, DesignRunError> {
    let source = antecedent_io::consume_analysis_result(bytes)
        .map_err(|_| RecalcRunError::Request("recalc.invalid_design_artifact"))?;
    let request = request.ok_or(RecalcRunError::Request("recalc.design_data_unavailable"))?;
    if !source.acceptance.recognized
        || !source.acceptance.claim_present
        || source.contract.is_none()
        || source.acceptance.unresolved.iter().any(|reason| {
            !matches!(request.model, DesignModel::Rd { .. })
                || reason.as_ref() != "dependencies.checked_rd_operation"
        })
    {
        return Err(RecalcRunError::Request("recalc.design_artifact_unverified").into());
    }
    let mut session = DesignSession::new();
    let outcome = execute_design_with_receipt(&mut session, request, ctx)?;
    let replayed = antecedent_io::consume_analysis_result(&session.export_result()?)
        .map_err(|_| RecalcRunError::Request("recalc.invalid_design_artifact"))?;
    // Compare the existing canonical wire digest, including unavailable numeric
    // representations, rather than floating-point PartialEq on NaN diagnostics.
    let normalize = |body: &antecedent_io::AnalysisResultWire| {
        antecedent_io::result_digest(body)
            .map_err(|_| RecalcRunError::Request("recalc.invalid_design_artifact"))
    };
    if source.header != replayed.header
        || source.contract != replayed.contract
        || normalize(&source.body)? != normalize(&replayed.body)?
    {
        return Err(RecalcRunError::Request("recalc.design_artifact_mismatch").into());
    }
    Ok(DesignReplay { session, outcome, artifact_digest: source.artifact_digest })
}

/// Execute checked existing engines; failures never replace the prior successful state.
/// # Errors
/// Invalid design, failed checked support, weak IV, unavailable fresh state, cancellation or
/// memory refusal, existing numerical failure, or inconsistent shared receipt.
pub fn execute_design_with_receipt(
    session: &mut DesignSession,
    r: &DesignRequest,
    ctx: &ExecutionContext,
) -> Result<RecalcOutcome, DesignRunError> {
    r.validate()?;
    let requested = r.identities(ctx);
    let plan = session.plan(r, ctx);
    if !plan.is_executable() {
        return Err(RecalcRunError::Refused(Box::new(plan)).into());
    }
    if ctx.cancellation.is_cancelled() {
        return Err(RecalcRunError::Request("recalc.cancelled").into());
    }
    let mut recorder = ReceiptRecorder::new();
    let held = session.live.as_ref();
    let (prepared, result, fit_ctx) = if recomputed(&plan, Stage::ScoreArtifact) {
        fit_guard(r, ctx)?;
        let (built, solves) = count_least_squares_solves(|| -> Result<_, DesignRunError> {
            let data = r.data()?;
            let (prepared, result) = if recomputed(&plan, Stage::Identification) {
                let prepared = r.study(data)?.prepare(ctx)?;
                r.check_design(&prepared)?;
                recorder.record(Counter::Identification, 1);
                let result = prepared.estimate_retained(ctx)?;
                (prepared, result)
            } else {
                let mut prepared = held
                    .ok_or(RecalcRunError::NoLiveState(Stage::Identification))?
                    .prepared
                    .as_ref()
                    .clone();
                let result = prepared.refresh(data, ctx)?;
                r.check_design(&prepared)?;
                (prepared, result)
            };
            if matches!(r.model, DesignModel::Iv2Sls { .. }) {
                let diagnostics = result
                    .estimate
                    .as_effect()
                    .and_then(|e| e.first_stage_diagnostics.as_ref())
                    .ok_or(RecalcRunError::Request("recalc.iv_diagnostics_unavailable"))?;
                if !diagnostics
                    .anderson_rubin
                    .is_some_and(|(lo, hi, _)| lo.is_finite() && hi.is_finite())
                {
                    return Err(DesignRunError::WeakInstrument {
                        point: result.effect(),
                        model_fits: 0,
                        diagnostics: Box::new(diagnostics.clone()),
                    });
                }
            }
            Ok((Arc::new(prepared), Arc::new(result), ctx.clone()))
        });
        recorder.record(Counter::ModelFit, solves);
        let built = built.map_err(|error| match error {
            DesignRunError::WeakInstrument { point, diagnostics, .. } => {
                DesignRunError::WeakInstrument { point, diagnostics, model_fits: solves }
            }
            other @ DesignRunError::Recalc(_) => other,
        });
        built?
    } else {
        let held = held.ok_or(RecalcRunError::NoLiveState(Stage::ScoreArtifact))?;
        (Arc::clone(&held.prepared), Arc::clone(&held.result), held.ctx.clone())
    };
    let law = if recomputed(&plan, Stage::Law) {
        summarize_checked_result(&result, &mut recorder)?
    } else {
        held.ok_or(RecalcRunError::NoLiveState(Stage::Law))?.law
    };
    let decision = if recomputed(&plan, Stage::Decision) {
        decide(law, r.utility, &mut recorder)?
    } else {
        held.ok_or(RecalcRunError::NoLiveState(Stage::Decision))?.decision
    };
    let receipt = recorder.finish(&plan).map_err(RecalcRunError::from)?;
    session.live = Some(Live { prepared, result, ctx: fit_ctx, law, decision });
    session.previous = requested;
    session.boundary = Boundary::InProcess;
    Ok(RecalcOutcome { plan, receipt, law, decision })
}
