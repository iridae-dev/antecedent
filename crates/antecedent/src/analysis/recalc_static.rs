//! Shared-plan execution of checked finite ADMG responses and multi-source transport.
//! Reused numerical factors carry no new uncertainty or causal identification license.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use super::recalc_receipt::{
    Counter, DecisionValue, LawValue, RecalcReceipt, RecalcRunError, ReceiptRecorder, UtilitySpec,
    decide,
};
use crate::{
    CausalError, EstimatorId, IdentifierId, MzTransportResult, PreparedMzTransport, PreparedStudy,
    RefuteSuite, Study, StudyBuilder, StudyResult,
};
use antecedent_core::recalc::{
    Boundary, RecalcCapabilities, RecalcPlan, RequestSupport, ResumeContext, RetargetSupport,
    Stage, StageIdentities, StageIdentity, StageStatus,
};
use antecedent_core::{
    ContinuousDomain, EvidenceCatalog, ExecutionContext, GridSpec, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseValue, SearchLimits, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::functional_distribution::EmpiricalFactorCache;
use antecedent_expr::execution_counts::{StaticWorkCounts, count_static_work};
use antecedent_expr::{Assignment, ExactEvaluationLimits, ExactTransportData};
use antecedent_graph::Admg;
use antecedent_identify::execution_counts::count_checked_identifications;
use antecedent_identify::{
    MzTransportDecision, MzTransportQuery, bind_mz_transport_catalog, decide_mz_transport,
};
use std::sync::Arc;
/// Executed finite point-law projection. Uncertainty remains in the original engine artifact.
#[derive(Debug)]
pub struct StaticRecalcOutcome {
    /// Shared dependency plan.
    pub plan: RecalcPlan,
    /// Actual executing component counts.
    pub receipt: RecalcReceipt,
    /// Ordered selected action means.
    pub means: Vec<f64>,
    /// Active minus baseline mean.
    pub contrast: f64,
    /// Point net-benefit decision.
    pub decision: DecisionValue,
}
/// Finite static ADMG mean-response request, identified through general ID.
#[derive(Clone, Debug)]
pub struct StaticResponseRequest {
    /// Numeric data/schema in stable graph order.
    pub columns: Vec<(String, Vec<f64>)>,
    /// Explicit static ADMG.
    pub graph: Admg,
    /// Intervention coordinate.
    pub treatment: VariableId,
    /// Scalar mean outcome.
    pub outcome: VariableId,
    /// Full checked finite action grid.
    pub support: Vec<f64>,
    /// Selected/reordered supported actions.
    pub actions: Vec<f64>,
    /// Selected baseline index.
    pub baseline: usize,
    /// Selected active index.
    pub active: usize,
    /// Decision rule.
    pub utility: UtilitySpec,
}
/// Actual raw multi-source query, evidence and finite law providers.
#[derive(Clone, Debug)]
pub struct MzRecalcRequest {
    /// Graph-order immutable names.
    pub variable_names: Vec<String>,
    /// Explicit static ADMG.
    pub graph: Admg,
    /// Checked by the executing bounded TR^mz procedure.
    pub query: MzTransportQuery,
    /// Evidence ownership and physical bindings.
    pub catalog: EvidenceCatalog,
    /// Exact or empirical finite provider laws.
    pub data: ExactTransportData,
    /// Joint action assignments in output order.
    pub requests: Vec<Assignment>,
    /// Theorem-specific bounded search limits.
    pub search: SearchLimits,
    /// Numerical evaluation bounds.
    pub limits: ExactEvaluationLimits,
    /// Baseline request index.
    pub baseline: usize,
    /// Active request index.
    pub active: usize,
    /// Decision rule.
    pub utility: UtilitySpec,
}
fn literal(label: &str, value: &str) -> StageIdentity {
    StageIdentity::of(label, &[value.as_bytes()])
}
fn float_bytes(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect()
}
fn graph_identity(graph: &Admg, names: &[String]) -> StageIdentity {
    let directed: Vec<_> = (0..graph.node_count())
        .flat_map(|i| {
            graph
                .parents(antecedent_graph::DenseNodeId::from_raw(
                    u32::try_from(i).unwrap_or(u32::MAX),
                ))
                .iter()
                .map(move |p| (p.raw(), i))
        })
        .collect();
    let bidirected: Vec<_> = (0..graph.node_count())
        .flat_map(|i| {
            graph
                .bidirected_neighbors(antecedent_graph::DenseNodeId::from_raw(
                    u32::try_from(i).unwrap_or(u32::MAX),
                ))
                .iter()
                .filter(move |b| b.raw() as usize > i)
                .map(move |b| (i, b.raw()))
        })
        .collect();
    literal(
        "static.graph.v1",
        &format!("{:?}:{directed:?}:{bidirected:?}:{names:?}", graph.nodes()),
    )
}
fn base_ids(graph: &Admg, names: &[String], query: &str) -> StageIdentities {
    let mut ids = StageIdentities::new();
    ids.set(Stage::Graph, graph_identity(graph, names));
    ids.set(Stage::Query, literal("static.query.v1", query));
    ids.set(Stage::Regime, literal("static.regime.v1", "explicit.finite"));
    ids.set(Stage::Evidence, literal("static.evidence.v1", "observational"));
    ids.set(Stage::SourcePopulation, literal("static.source.v1", "observed"));
    ids.set(Stage::TargetPopulation, literal("static.target.v1", "all_observed"));
    ids.set(Stage::RowDesign, literal("static.rows.v1", "joint_complete_case"));
    ids.set(
        Stage::LearnerFoldsRng,
        literal("static.inference.v1", "frequentist.point_only.no_resampling"),
    );
    ids.set(Stage::Identification, literal("static.checked.v1", "generic_id_or_bounded_trmz"));
    ids.set(Stage::ScoreArtifact, literal("static.artifact.v1", "proof_bound_factors"));
    ids.set(Stage::Decision, literal("static.decision.v1", "net_benefit"));
    ids
}
fn changed(plan: &RecalcPlan, stage: Stage) -> bool {
    matches!(plan.status(stage), Some(StageStatus::Recomputed { .. }))
}
fn scope_plan(
    previous: &StageIdentities,
    requested: &StageIdentities,
    boundary: Boundary,
    supported: bool,
) -> RecalcPlan {
    RecalcPlan::plan(
        previous,
        requested,
        &RecalcCapabilities {
            retarget: RetargetSupport::NotDeclared,
            request: if supported {
                RequestSupport::OnGrid
            } else {
                RequestSupport::OffGrid { licensed_route: None }
            },
            boundary,
        },
    )
}
fn boundary(resume: ResumeContext) -> Boundary {
    Boundary::FreshProcess(ResumeContext {
        supplied_data: resume.supplied_data,
        ..ResumeContext::default()
    })
}
fn io_error(error: antecedent_io::IoError) -> RecalcRunError {
    RecalcRunError::Execution(CausalError::Serialization(error))
}
fn guards(ctx: &ExecutionContext, values: usize) -> Result<(), RecalcRunError> {
    if ctx.cancellation.is_cancelled() {
        return Err(RecalcRunError::Request("recalc.cancelled"));
    }
    let bytes = (values as u64).saturating_mul(128);
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
fn names_valid(names: &[String], n: usize) -> bool {
    names.len() == n
        && n > 0
        && n <= 12
        && names.iter().all(|name| !name.is_empty())
        && names.iter().collect::<std::collections::BTreeSet<_>>().len() == n
}
fn contrast(means: &[f64], baseline: usize, active: usize) -> Result<f64, RecalcRunError> {
    let (Some(a), Some(b)) = (means.get(active), means.get(baseline)) else {
        return Err(RecalcRunError::Request("recalc.static_invalid_contrast"));
    };
    let value = a - b;
    if !value.is_finite() {
        return Err(RecalcRunError::Request("recalc.static_required_factor_missing"));
    }
    Ok(value)
}
fn summarize_contrast(
    means: &[f64],
    baseline: usize,
    active: usize,
) -> Result<f64, RecalcRunError> {
    let value = contrast(means, baseline, active)?;
    antecedent_expr::execution_counts::note_static_work(
        antecedent_expr::execution_counts::StaticWork::LawSummary,
    );
    Ok(value)
}
fn record_artifact(recorder: &mut ReceiptRecorder, work: StaticWorkCounts) {
    recorder.record(Counter::FactorBuild, work.factor_builds);
    recorder.record(Counter::ProgramCompilation, work.program_compilations);
    recorder.record(Counter::ProviderBinding, work.provider_bindings);
}
fn record_law(recorder: &mut ReceiptRecorder, work: StaticWorkCounts) {
    recorder.record(Counter::LawSummary, work.law_summaries);
    recorder.record(Counter::FactorEvaluation, work.factor_evaluations);
    recorder.record(Counter::Integration, work.integrations);
    recorder.record(Counter::ProviderCall, work.provider_calls);
}
impl StaticResponseRequest {
    /// Semantic identities include checked support separately from selected reductions.
    #[must_use]
    pub fn identities(&self) -> StageIdentities {
        let names = self.columns.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>();
        let mut ids = base_ids(
            &self.graph,
            &names,
            &format!("response:{}:{}:{:?}", self.treatment.raw(), self.outcome.raw(), self.support),
        );
        let mut parts = Vec::new();
        for (name, col) in &self.columns {
            parts.push(name.as_bytes().to_vec());
            parts.push(float_bytes(col));
        }
        ids.set(
            Stage::DataSnapshot,
            StageIdentity::of(
                "static.snapshot.v1",
                &parts.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            ),
        );
        ids.set(
            Stage::TreatmentGrid,
            StageIdentity::of("static.support.v1", &[&float_bytes(&self.support)]),
        );
        ids.set(Stage::Utility, super::recalc_cell::utility_identity(self.utility));
        ids.set(
            Stage::Law,
            literal(
                "static.selection.v1",
                &format!("{:?}:{}:{}", self.actions, self.baseline, self.active),
            ),
        );
        ids
    }
    fn supported(&self) -> bool {
        self.graph.has_bidirected()
            && self
                .actions
                .iter()
                .all(|action| self.support.iter().any(|v| v.to_bits() == action.to_bits()))
            && self.support.iter().all(|value| {
                self.columns
                    .get(self.treatment.raw() as usize)
                    .is_some_and(|(_, col)| col.contains(value))
            })
    }
    fn validate(&self) -> Result<(), RecalcRunError> {
        self.utility.validate()?;
        if !self.graph.has_bidirected() {
            return Err(RecalcRunError::Request("recalc.static_graph_unsupported"));
        }
        let names = self.columns.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>();
        let n = self.columns.first().map_or(0, |(_, v)| v.len());
        if !names_valid(&names, self.graph.node_count())
            || n == 0
            || n > 100_000
            || n.saturating_mul(names.len()) > 1_000_000
            || self.columns.iter().any(|(_, values)| {
                values.len() != n
                    || values.iter().any(|v| v.is_infinite())
                    || values
                        .iter()
                        .filter(|v| v.is_finite())
                        .map(|v| v.to_bits())
                        .collect::<std::collections::BTreeSet<_>>()
                        .len()
                        > 64
            })
        {
            return Err(RecalcRunError::Request("recalc.static_invalid_data"));
        }
        if self.treatment == self.outcome
            || self.treatment.raw() as usize >= names.len()
            || self.outcome.raw() as usize >= names.len()
            || self.support.len() < 2
            || self.support.len() > 64
            || self.support.iter().any(|v| !v.is_finite())
            || self.support.windows(2).any(|pair| pair[0] >= pair[1])
            || self
                .support
                .iter()
                .map(|v| v.to_bits())
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != self.support.len()
            || self.actions.is_empty()
            || self.actions.len() > 64
            || self.actions.iter().any(|v| !v.is_finite())
        {
            return Err(RecalcRunError::Request("recalc.static_invalid_query"));
        }
        if self
            .actions
            .iter()
            .any(|action| !self.support.iter().any(|value| value.to_bits() == action.to_bits()))
        {
            return Err(RecalcRunError::Request("recalc.static_action_out_of_support"));
        }
        if self.baseline >= self.actions.len() || self.active >= self.actions.len() {
            return Err(RecalcRunError::Request("recalc.static_invalid_contrast"));
        }
        Ok(())
    }
    fn factor_workspace_values(&self) -> usize {
        // Disjoint variable/conditioning sets have three states per column.
        // Budget every possible descriptor and checked response member before
        // allocation; actual ID may need fewer, but never a larger domain.
        let cells = self.columns.iter().fold(1usize, |cells, (_, values)| {
            let levels = values
                .iter()
                .filter(|v| v.is_finite())
                .map(|v| v.to_bits())
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            cells.saturating_mul(levels.max(1))
        });
        let descriptors = (0..self.columns.len()).fold(1usize, |n, _| n.saturating_mul(3));
        cells
            .saturating_mul(descriptors)
            .saturating_mul(self.support.len())
            .saturating_mul(self.columns.len().saturating_mul(2).saturating_add(1))
    }
    fn data(&self) -> Result<TabularData, RecalcRunError> {
        TabularData::from_f64_columns(
            self.columns.iter().map(|(n, v)| (n.as_str(), v.as_slice())).collect::<Vec<_>>(),
        )
        .map_err(|e| RecalcRunError::Execution(e.into()))
    }
    fn study(&self) -> Result<Study, RecalcRunError> {
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: self.outcome,
            treatment: ContinuousDomain::new(
                self.treatment,
                GridSpec::Values(self.support.clone().into()),
            ),
        });
        Ok(Study::tabular(self.data()?)
            .graph(self.graph.clone())
            .query(query)
            .identifier(IdentifierId::GeneralId)
            .estimator(EstimatorId::FunctionalEffect)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()?)
    }
}
#[derive(Clone)]
struct StaticLive {
    prepared: Arc<PreparedStudy>,
    result: Arc<StudyResult>,
    full_means: Vec<f64>,
    means: Vec<f64>,
    contrast: f64,
    decision: DecisionValue,
    context: ExecutionContext,
    producing_receipt: Arc<RecalcReceipt>,
}
/// Retains checked programs and exact empirical factors; flags cannot create state.
#[derive(Clone)]
pub struct StaticResponseSession {
    previous: StageIdentities,
    live: Option<StaticLive>,
    boundary: Boundary,
    cache: Arc<EmpiricalFactorCache>,
}
impl Default for StaticResponseSession {
    fn default() -> Self {
        Self::new()
    }
}
impl StaticResponseSession {
    /// Empty session.
    #[must_use]
    pub fn new() -> Self {
        Self {
            previous: StageIdentities::new(),
            live: None,
            boundary: Boundary::InProcess,
            cache: EmpiricalFactorCache::new(),
        }
    }
    /// Identities-only resume, requiring real supplied data to reconstruct factors.
    #[must_use]
    pub fn resume(previous: StageIdentities, resume: ResumeContext) -> Self {
        Self {
            previous,
            live: None,
            boundary: boundary(resume),
            cache: EmpiricalFactorCache::new(),
        }
    }
    /// Last successful semantic identities.
    #[must_use]
    pub const fn identities(&self) -> &StageIdentities {
        &self.previous
    }
    /// Actual resume boundary.
    #[must_use]
    pub const fn boundary(&self) -> Boundary {
        self.boundary
    }
    /// Actual retained checked state.
    #[must_use]
    pub const fn is_live(&self) -> bool {
        self.live.is_some()
    }
    /// Genuine retained numeric fit identities.
    #[must_use]
    pub fn factor_identities(&self) -> Vec<StageIdentity> {
        self.cache.identities()
    }
    /// Shared plan.
    #[must_use]
    pub fn plan(&self, request: &StaticResponseRequest, _ctx: &ExecutionContext) -> RecalcPlan {
        scope_plan(&self.previous, &request.identities(), self.boundary, request.supported())
    }
    /// Original native-issued full-support scientific result.
    #[must_use]
    pub fn result(&self) -> Option<&StudyResult> {
        self.live.as_ref().map(|live| live.result.as_ref())
    }
    /// Original retained checked preparation; borrowing it performs no identification or fit.
    #[must_use]
    pub fn prepared(&self) -> Option<&PreparedStudy> {
        self.live.as_ref().map(|live| live.prepared.as_ref())
    }
    /// Receipt of the actual retained factor/result production, preserved across projection reuse.
    #[must_use]
    pub fn producing_receipt(&self) -> Option<&RecalcReceipt> {
        self.live.as_ref().map(|live| live.producing_receipt.as_ref())
    }
    /// Exact retained graph-order schema used by checked execution.
    #[must_use]
    pub fn schema(&self) -> Option<&antecedent_core::CausalSchema> {
        let live = self.live.as_ref()?;
        let super::builder::DataInput::Tabular(data) = &live.prepared.study().data else {
            return None;
        };
        Some(data.schema())
    }
    /// Actual original producing context; projection changes do not rewrite it.
    #[must_use]
    pub fn producing_context(&self) -> Option<&ExecutionContext> {
        self.live.as_ref().map(|live| &live.context)
    }
    /// Export the original checked response result under its producing context.
    /// # Errors
    /// No live state or a refused result binding.
    pub fn export_result(&self, ctx: &ExecutionContext) -> Result<Vec<u8>, RecalcRunError> {
        guards(ctx, 0)?;
        let live = self.live.as_ref().ok_or(RecalcRunError::NoLiveState(Stage::Law))?;
        Ok(live.prepared.encode_contracted_result(
            &live.result,
            "recalc-static-response",
            &live.context,
        )?)
    }
}
fn response_means(result: &StudyResult) -> Result<Vec<f64>, RecalcRunError> {
    match result.response.as_ref().map(|r| &r.estimate) {
        Some(ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }))
            if mean.iter().all(|v| v.is_finite()) =>
        {
            Ok(mean.to_vec())
        }
        _ => Err(RecalcRunError::Request("recalc.static_required_factor_missing")),
    }
}
/// Execute static checked preparation, selective factor rebinding and point-law reductions.
/// # Errors
/// Typed schema, support, identification/provider, cancellation/budget and receipt refusals.
pub fn execute_static_response_with_receipt(
    session: &mut StaticResponseSession,
    request: &StaticResponseRequest,
    ctx: &ExecutionContext,
) -> Result<StaticRecalcOutcome, RecalcRunError> {
    request.validate()?;
    let ids = request.identities();
    let plan = scope_plan(&session.previous, &ids, session.boundary, request.supported());
    if !plan.is_executable() {
        return Err(RecalcRunError::Refused(Box::new(plan)));
    }
    if ctx.cancellation.is_cancelled() {
        return Err(RecalcRunError::Request("recalc.cancelled"));
    }
    let mut recorder = ReceiptRecorder::new();
    let old = session.live.as_ref();
    let prepared = if changed(&plan, Stage::ScoreArtifact) {
        guards(
            ctx,
            request
                .columns
                .iter()
                .map(|(_, v)| v.len())
                .sum::<usize>()
                .saturating_add(request.factor_workspace_values()),
        )?;
        let ((prepared, checks), work) = count_static_work(|| {
            count_checked_identifications(|| {
                session.cache.scope(|| -> Result<PreparedStudy, RecalcRunError> {
                    if changed(&plan, Stage::Identification) {
                        request.study()?.prepare(ctx).map_err(Into::into)
                    } else {
                        let mut p = old
                            .ok_or(RecalcRunError::NoLiveState(Stage::Identification))?
                            .prepared
                            .as_ref()
                            .clone();
                        p.rebind_static_functional_data(request.data()?)?;
                        Ok(p)
                    }
                })
            })
        });
        recorder.record(Counter::Identification, checks);
        record_artifact(&mut recorder, work);
        Arc::new(prepared?)
    } else {
        Arc::clone(&old.ok_or(RecalcRunError::NoLiveState(Stage::ScoreArtifact))?.prepared)
    };
    let (result, full_means, means, value, context) = if changed(&plan, Stage::Law) {
        let must_evaluate = changed(&plan, Stage::ScoreArtifact);
        let (result, full_means, context) = if must_evaluate {
            let (result, work) =
                count_static_work(|| prepared.execute_static_functional_retained(ctx));
            record_law(&mut recorder, work);
            let result = Arc::new(result?);
            let full = response_means(&result)?;
            (result, full, ctx.clone())
        } else {
            let old = old.ok_or(RecalcRunError::NoLiveState(Stage::Law))?;
            (Arc::clone(&old.result), old.full_means.clone(), old.context.clone())
        };
        let means = request
            .actions
            .iter()
            .map(|action| {
                request
                    .support
                    .iter()
                    .position(|v| v.to_bits() == action.to_bits())
                    .and_then(|i| full_means.get(i).copied())
                    .ok_or(RecalcRunError::Request("recalc.static_action_out_of_support"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let value = if must_evaluate {
            contrast(&means, request.baseline, request.active)?
        } else {
            let (value, work) =
                count_static_work(|| summarize_contrast(&means, request.baseline, request.active));
            record_law(&mut recorder, work);
            value?
        };
        (result, full_means, means, value, context)
    } else {
        let old = old.ok_or(RecalcRunError::NoLiveState(Stage::Law))?;
        (
            Arc::clone(&old.result),
            old.full_means.clone(),
            old.means.clone(),
            old.contrast,
            old.context.clone(),
        )
    };
    let decision = if changed(&plan, Stage::Decision) {
        decide(LawValue { ate: value, std_error: f64::NAN }, request.utility, &mut recorder)?
    } else {
        old.ok_or(RecalcRunError::NoLiveState(Stage::Decision))?.decision
    };
    let receipt = recorder.finish(&plan)?;
    let producing_receipt = if changed(&plan, Stage::ScoreArtifact) {
        Arc::new(receipt.clone())
    } else {
        Arc::clone(&old.ok_or(RecalcRunError::NoLiveState(Stage::ScoreArtifact))?.producing_receipt)
    };
    session.live = Some(StaticLive {
        prepared,
        result,
        full_means,
        means: means.clone(),
        contrast: value,
        decision,
        context,
        producing_receipt,
    });
    session.previous = ids;
    session.boundary = Boundary::InProcess;
    Ok(StaticRecalcOutcome { plan, receipt, means, contrast: value, decision })
}
impl MzRecalcRequest {
    /// Complete query/catalog/law and finite assignment identities.
    #[must_use]
    pub fn identities(&self) -> StageIdentities {
        let mut ids = base_ids(
            &self.graph,
            &self.variable_names,
            &format!("mz:{:?}", self.query.canonical()),
        );
        ids.set(
            Stage::Evidence,
            literal("mz.catalog.v1", &format!("{:?}:{:?}", self.catalog, self.search)),
        );
        ids.set(Stage::Regime, literal("mz.regimes.v1", &format!("{:?}", self.catalog.regimes)));
        ids.set(
            Stage::SourcePopulation,
            literal("mz.sources.v1", &format!("{:?}", self.query.canonical().sources)),
        );
        let source_ids = self
            .data
            .law_identities()
            .iter()
            .map(|id| id.as_bytes().as_slice())
            .collect::<Vec<_>>();
        ids.set(Stage::DataSnapshot, StageIdentity::of("mz.laws.v1", &source_ids));
        let target_ids = self
            .data
            .laws()
            .iter()
            .filter(|law| law.population() == self.query.target.as_ref())
            .map(antecedent_expr::ExactDiscreteLaw::content_identity)
            .collect::<Vec<_>>();
        ids.set(
            Stage::TargetPopulation,
            StageIdentity::of(
                "mz.target.v1",
                &target_ids.iter().map(|id| id.as_bytes().as_slice()).collect::<Vec<_>>(),
            ),
        );
        ids.set(Stage::RowDesign, literal("mz.rows.v1", "finite.law.axes.and.proof_leaves"));
        ids.set(Stage::TreatmentGrid, literal("mz.actions.v1", &format!("{:?}", self.requests)));
        ids.set(Stage::LearnerFoldsRng, literal("mz.evaluation.v1", &format!("{:?}", self.limits)));
        ids.set(
            Stage::Law,
            literal("mz.reduction.v1", &format!("{}:{}", self.baseline, self.active)),
        );
        ids.set(Stage::Utility, super::recalc_cell::utility_identity(self.utility));
        ids
    }
    fn supported(&self) -> bool {
        self.requests.iter().all(|request| {
            self.query.treatments.iter().all(|&treatment| {
                request.get(treatment).is_some_and(|value| {
                    self.data.domain(treatment).is_some_and(|domain| domain.contains(value))
                })
            })
        })
    }
    fn validate(&self) -> Result<(), RecalcRunError> {
        self.utility.validate()?;
        if !names_valid(&self.variable_names, self.graph.node_count())
            || self.query.outcomes.len() != 1
            || self.requests.is_empty()
            || self.requests.len() > 64
            || self
                .requests
                .iter()
                .any(|request| request.entries().len() != self.query.treatments.len())
            || self.data.laws().len() > 64
            || self.data.laws().is_empty()
            || self
                .data
                .laws()
                .iter()
                .any(|law| law.axes().iter().any(|axis| axis.values.len() > 64))
            || self.data.laws().iter().map(|law| law.probabilities().len()).sum::<usize>()
                > 1_000_000
            || self.baseline >= self.requests.len()
            || self.active >= self.requests.len()
        {
            return Err(RecalcRunError::Request("recalc.mz_invalid_request"));
        }
        Ok(())
    }
}
struct MzLive {
    prepared: Arc<PreparedMzTransport>,
    result: Arc<MzTransportResult>,
    means: Vec<f64>,
    contrast: f64,
    decision: DecisionValue,
    names: Vec<String>,
    context: ExecutionContext,
}
/// Real retained multi-source proof, provider cache and compiled plans.
pub struct MzRecalcSession {
    previous: StageIdentities,
    live: Option<MzLive>,
    boundary: Boundary,
}
impl Default for MzRecalcSession {
    fn default() -> Self {
        Self::new()
    }
}
impl MzRecalcSession {
    /// Empty session.
    #[must_use]
    pub fn new() -> Self {
        Self { previous: StageIdentities::new(), live: None, boundary: Boundary::InProcess }
    }
    /// Identities-only resume; raw evidence/laws must actually be supplied to rebuild.
    #[must_use]
    pub fn resume(previous: StageIdentities, resume: ResumeContext) -> Self {
        Self { previous, live: None, boundary: boundary(resume) }
    }
    /// Last successful semantic identities.
    #[must_use]
    pub const fn identities(&self) -> &StageIdentities {
        &self.previous
    }
    /// Actual resume boundary.
    #[must_use]
    pub const fn boundary(&self) -> Boundary {
        self.boundary
    }
    /// Actual retained checked state.
    #[must_use]
    pub const fn is_live(&self) -> bool {
        self.live.is_some()
    }
    /// Numeric law identities retain population/regime/world/snapshot/probability/count binding.
    #[must_use]
    pub fn factor_identities(&self) -> Vec<StageIdentity> {
        self.live
            .as_ref()
            .map_or_else(Vec::new, |live| live.prepared.data().law_identities().to_vec())
    }
    /// Shared executable/refusal plan.
    #[must_use]
    pub fn plan(&self, request: &MzRecalcRequest, _ctx: &ExecutionContext) -> RecalcPlan {
        scope_plan(&self.previous, &request.identities(), self.boundary, request.supported())
    }
    /// Export original checked source bindings and evaluated target laws under producing context.
    /// # Errors
    /// Missing state or a refused exact result/proof/names binding.
    pub fn export_result(&self, ctx: &ExecutionContext) -> Result<Vec<u8>, RecalcRunError> {
        guards(ctx, 0)?;
        let live = self.live.as_ref().ok_or(RecalcRunError::NoLiveState(Stage::Law))?;
        live.result.export_named(&live.prepared, &live.names, &live.context).map_err(io_error)
    }
}
/// Execute a genuine bounded multi-source decision, retained provider refresh, and finite laws.
/// # Errors
/// Typed theorem, missing-factor, support, search/budget, cancellation and receipt refusals.
pub fn execute_mz_with_receipt(
    session: &mut MzRecalcSession,
    request: &MzRecalcRequest,
    ctx: &ExecutionContext,
) -> Result<StaticRecalcOutcome, RecalcRunError> {
    request.validate()?;
    let ids = request.identities();
    let plan = scope_plan(&session.previous, &ids, session.boundary, request.supported());
    if !plan.is_executable() {
        return Err(RecalcRunError::Refused(Box::new(plan)));
    }
    if ctx.cancellation.is_cancelled() {
        return Err(RecalcRunError::Request("recalc.cancelled"));
    }
    let old = session.live.as_ref();
    let mut recorder = ReceiptRecorder::new();
    let prepared = if changed(&plan, Stage::ScoreArtifact) {
        guards(
            ctx,
            request
                .data
                .laws()
                .iter()
                .map(|law| law.probabilities().len().saturating_mul(law.axes().len()))
                .sum(),
        )?;
        let ((prepared, checks), work) = count_static_work(|| {
            count_checked_identifications(|| -> Result<PreparedMzTransport, RecalcRunError> {
                let data = request
                    .data
                    .rebind_reusing_factor_cache(old.map(|old| old.prepared.data()), 4096)
                    .map_err(|e| {
                        io_error(antecedent_io::IoError::Refused {
                            code: antecedent_core::reason_code!("invalid_argument"),
                            message: e.to_string(),
                        })
                    })?;
                if changed(&plan, Stage::Identification) {
                    let decision = decide_mz_transport(
                        &request.graph,
                        &request.query,
                        &request.catalog,
                        request.search,
                        ctx,
                    )
                    .map_err(|error| {
                        io_error(antecedent_io::mz_transport_artifact::mz_identification_error(
                            error,
                        ))
                    })?;
                    let MzTransportDecision::Identified { derivation, .. } = decision else {
                        return Err(io_error(antecedent_io::IoError::Refused {
                            code: decision.reason_code().unwrap_or("transport_not_certified"),
                            message: decision
                                .detail_code()
                                .unwrap_or("mz_transport.search_incomplete")
                                .to_owned(),
                        }));
                    };
                    let functional =
                        bind_mz_transport_catalog(&request.graph, &derivation, &request.catalog)
                            .map_err(|error| {
                                io_error(
                                    antecedent_io::mz_transport_artifact::mz_identification_error(
                                        error,
                                    ),
                                )
                            })?;
                    StudyBuilder::mz_transport(
                        request.graph.clone(),
                        functional,
                        request.search,
                        data,
                        request.requests.clone(),
                        request.limits,
                        ctx,
                    )
                    .map_err(io_error)
                } else {
                    old.ok_or(RecalcRunError::NoLiveState(Stage::Identification))?
                        .prepared
                        .refresh_requests(data, request.requests.clone(), request.limits, ctx)
                        .map_err(io_error)
                }
            })
        });
        recorder.record(Counter::Identification, checks);
        record_artifact(&mut recorder, work);
        Arc::new(prepared?)
    } else {
        Arc::clone(&old.ok_or(RecalcRunError::NoLiveState(Stage::ScoreArtifact))?.prepared)
    };
    let (result, means, value, context) = if changed(&plan, Stage::Law) {
        if changed(&plan, Stage::ScoreArtifact) {
            let (evaluated, work) = count_static_work(|| -> Result<_, RecalcRunError> {
                let result = prepared.estimate(ctx).map_err(io_error)?;
                let means = result
                    .distributions()
                    .iter()
                    .map(|law| {
                        law.mean(request.query.outcomes[0]).map_err(|error| {
                            io_error(antecedent_io::IoError::Refused {
                                code: antecedent_core::reason_code!(
                                    "transport_unsupported_evaluator"
                                ),
                                message: error.to_string(),
                            })
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((Arc::new(result), means))
            });
            record_law(&mut recorder, work);
            let (result, means) = evaluated?;
            let value = contrast(&means, request.baseline, request.active)?;
            (result, means, value, ctx.clone())
        } else {
            let old = old.ok_or(RecalcRunError::NoLiveState(Stage::Law))?;
            let means = old.means.clone();
            let (value, work) =
                count_static_work(|| summarize_contrast(&means, request.baseline, request.active));
            record_law(&mut recorder, work);
            let value = value?;
            (Arc::clone(&old.result), means, value, old.context.clone())
        }
    } else {
        let old = old.ok_or(RecalcRunError::NoLiveState(Stage::Law))?;
        (Arc::clone(&old.result), old.means.clone(), old.contrast, old.context.clone())
    };
    let decision = if changed(&plan, Stage::Decision) {
        decide(LawValue { ate: value, std_error: f64::NAN }, request.utility, &mut recorder)?
    } else {
        old.ok_or(RecalcRunError::NoLiveState(Stage::Decision))?.decision
    };
    let receipt = recorder.finish(&plan)?;
    session.live = Some(MzLive {
        prepared,
        result,
        means: means.clone(),
        contrast: value,
        decision,
        names: request.variable_names.clone(),
        context,
    });
    session.previous = ids;
    session.boundary = Boundary::InProcess;
    Ok(StaticRecalcOutcome { plan, receipt, means, contrast: value, decision })
}
