//! Shared-plan recalculation over the existing checked DML-AIPW and DR-Learner engines.
//! Model-point CATE prediction is distinct from marginal score inference.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use super::PreparedStudy;
use super::recalc_receipt::{
    Counter, DecisionValue, LawValue, RecalcOutcome, RecalcRunError, ReceiptRecorder,
    TargetWeights, UtilitySpec, decide,
};
use crate::estimator_spec::EstimatorSpec;
use crate::graph::{Dag, DenseNodeId};
use crate::{RefuteSuite, Study, StudyResult};
use antecedent_core::recalc::{
    Boundary, RecalcCapabilities, RecalcPlan, RequestSupport, ResumeContext, RetargetSupport,
    Stage, StageIdentities, StageIdentity, StageStatus,
};
use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::dml::{DmlScore, count_aipw_score_builds};
use antecedent_estimate::dr::count_dr_final_fits;
use antecedent_estimate::{
    CrossfitNuisanceCache, DmlAte, DrLearner, FittedEffect, OverlapPolicy, ScoreTable,
};
use antecedent_io::frozen_scores_artifact::FrozenScoreTable;
use antecedent_learn::LearnerSpec;
use antecedent_learn::fit_counts::count_resolved_fits;
use std::collections::BTreeSet;
use std::sync::Arc;

/// Existing estimator configuration. PLR and trimmed score reuse are explicitly unlicensed.
#[derive(Clone, Debug)]
pub enum DrEstimator {
    /// DML AIPW marginal score contract.
    Dml(DmlAte),
    /// DR-Learner marginal scores and retained CATE predictor.
    Cate(DrLearner),
}
impl DrEstimator {
    fn spec(&self) -> EstimatorSpec {
        match self {
            Self::Dml(config) => config.clone().into(),
            Self::Cate(config) => config.clone().into(),
        }
    }
    fn family(&self) -> &'static str {
        match self {
            Self::Dml(_) => "dml_aipw",
            Self::Cate(_) => "dr_cate",
        }
    }
    fn supported(&self) -> bool {
        let continuous = |s: LearnerSpec| {
            matches!(s, LearnerSpec::Linear(_) | LearnerSpec::Ridge(_)) && s.validate().is_ok()
        };
        let treatment = |s: LearnerSpec| {
            matches!(s, LearnerSpec::Linear(_) | LearnerSpec::Ridge(_) | LearnerSpec::Logistic(_))
                && s.validate().is_ok()
        };
        match self {
            Self::Dml(config) => {
                config.folds >= 2
                    && config.overlap.validate().is_ok()
                    && continuous(config.outcome)
                    && treatment(config.treatment)
                    && config.score == DmlScore::Aipw
                    && !matches!(
                        config.overlap,
                        OverlapPolicy::RequireDiagnostics { trim: Some(_), .. }
                    )
            }
            Self::Cate(config) => {
                config.folds >= 2
                    && config.overlap.validate().is_ok()
                    && continuous(config.outcome)
                    && treatment(config.treatment)
                    && continuous(config.final_learner)
                    && !matches!(
                        config.overlap,
                        OverlapPolicy::RequireDiagnostics { trim: Some(_), .. }
                    )
            }
        }
    }
}
/// Binary average-effect request with automatic checked graph adjustment and complete-case rows.
#[derive(Clone, Debug)]
pub struct DrRequest {
    /// Stable variable ids are positions in this schema.
    pub columns: Vec<(String, Vec<f64>)>,
    /// Directed edges in stable id space.
    pub edges: Vec<(u32, u32)>,
    /// Binary treatment role.
    pub treatment: u32,
    /// Outcome role.
    pub outcome: u32,
    /// Exact nuisance/final learner, folds and overlap configuration.
    pub estimator: DrEstimator,
    /// Target weights in the retained score row order, not raw frame order.
    pub target: Option<TargetWeights>,
    /// Net benefit rule.
    pub utility: UtilitySpec,
}
fn literal(label: &str, value: &str) -> StageIdentity {
    StageIdentity::of(label, &[value.as_bytes()])
}
fn floats(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect()
}
impl DrRequest {
    /// Own identity of every causal, physical and decision stage.
    #[must_use]
    pub fn identities(&self, ctx: &ExecutionContext) -> StageIdentities {
        let mut ids = StageIdentities::new();
        ids.set(Stage::Graph, super::recalc_cell::graph_identity(self.columns.len(), &self.edges));
        ids.set(
            Stage::Query,
            literal(
                "dr.query",
                &format!(
                    "binary.mean:{}:{}:{}",
                    self.treatment,
                    self.outcome,
                    self.estimator.family()
                ),
            ),
        );
        ids.set(Stage::Regime, literal("dr.regime", "static.binary.0_1"));
        ids.set(Stage::Evidence, literal("dr.evidence", "observational.no_external"));
        ids.set(Stage::SourcePopulation, literal("dr.source", "all_observed"));
        ids.set(Stage::TargetPopulation, super::recalc_cell::target_identity(self.target.as_ref()));
        let mut parts = Vec::new();
        for (name, values) in &self.columns {
            parts.push(name.as_bytes().to_vec());
            parts.push(floats(values));
        }
        ids.set(
            Stage::DataSnapshot,
            StageIdentity::of("dr.snapshot", &parts.iter().map(Vec::as_slice).collect::<Vec<_>>()),
        );
        ids.set(
            Stage::RowDesign,
            literal(
                "dr.rows",
                &format!("complete_case.raw={}", self.columns.first().map_or(0, |(_, v)| v.len())),
            ),
        );
        ids.set(Stage::TreatmentGrid, literal("dr.grid", "binary.0_1"));
        ids.set(
            Stage::LearnerFoldsRng,
            StageIdentity::of(
                "dr.config",
                &[format!("{:?}", self.estimator).as_bytes(), &ctx.rng.master_seed().to_le_bytes()],
            ),
        );
        ids.set(Stage::Utility, super::recalc_cell::utility_identity(self.utility));
        ids.set(
            Stage::Identification,
            literal("dr.identification", "checked_study.average_effect"),
        );
        ids.set(
            Stage::ScoreArtifact,
            literal("dr.fit", "complete_case.marginal_scores.retained_cate.v1"),
        );
        ids.set(Stage::Law, literal("dr.law", "weighted_binary_average_effect"));
        ids.set(Stage::Decision, literal("dr.decision", "net_benefit"));
        ids
    }
    fn graph(&self) -> Result<Dag, RecalcRunError> {
        let n = u32::try_from(self.columns.len())
            .map_err(|_| RecalcRunError::Request("recalc.invalid_graph"))?;
        let mut graph = Dag::with_variables(n);
        let mut edges = self.edges.clone();
        edges.sort_unstable();
        edges.dedup();
        for (a, b) in edges {
            graph
                .insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b))
                .map_err(|_| RecalcRunError::Request("recalc.invalid_graph"))?;
        }
        Ok(graph)
    }
    fn data(&self) -> Result<TabularData, RecalcRunError> {
        TabularData::from_f64_columns(
            self.columns
                .iter()
                .map(|(name, values)| (name.as_str(), values.as_slice()))
                .collect::<Vec<(&str, &[f64])>>(),
        )
        .map_err(|_| RecalcRunError::Request("recalc.invalid_data"))
    }
    fn study(&self, data: TabularData) -> Result<Study, RecalcRunError> {
        Ok(Study::tabular(data)
            .graph(self.graph()?)
            .query(AverageEffectQuery::binary_ate(
                VariableId::from_raw(self.treatment),
                VariableId::from_raw(self.outcome),
            ))
            .estimator(self.estimator.spec())
            .refute(RefuteSuite::None)
            .build()?)
    }
    fn validate(&self) -> Result<(), RecalcRunError> {
        self.utility.validate()?;
        let rows = self.columns.first().map_or(0, |(_, v)| v.len());
        let names: BTreeSet<_> = self.columns.iter().map(|(name, _)| name).collect();
        if rows == 0
            || names.len() != self.columns.len()
            || self.columns.iter().any(|(name, v)| {
                name.is_empty() || v.len() != rows || v.iter().any(|x| x.is_infinite())
            })
        {
            return Err(RecalcRunError::Request("recalc.invalid_data"));
        }
        if self.treatment == self.outcome
            || self.treatment as usize >= self.columns.len()
            || self.outcome as usize >= self.columns.len()
        {
            return Err(RecalcRunError::Request("recalc.invalid_query"));
        }
        if let Some(target) = &self.target {
            let mass = target.weights.iter().sum::<f64>();
            if target.weights.is_empty()
                || target.weights.iter().any(|x| !x.is_finite() || *x < 0.)
                || !mass.is_finite()
                || mass <= 0.
                || target.depends_on.iter().any(|id| {
                    id.raw() == self.treatment
                        || id.raw() == self.outcome
                        || id.raw() as usize >= self.columns.len()
                })
            {
                return Err(RecalcRunError::Request("recalc.invalid_target_weights"));
            }
        }
        Ok(())
    }
}
struct Live {
    prepared: Arc<PreparedStudy>,
    result: Arc<StudyResult>,
    fit_context: ExecutionContext,
    law: LawValue,
    decision: DecisionValue,
    bounds: Vec<(VariableId, f64, f64)>,
}
/// Holds checked prepared computation, scores and an actual optional CATE predictor.
pub struct DrSession {
    previous: StageIdentities,
    live: Option<Live>,
    boundary: Boundary,
    input_rows: u64,
}
impl Default for DrSession {
    fn default() -> Self {
        Self::new()
    }
}
impl DrSession {
    /// Empty in-process workflow.
    #[must_use]
    pub fn new() -> Self {
        Self {
            previous: StageIdentities::new(),
            live: None,
            boundary: Boundary::InProcess,
            input_rows: 0,
        }
    }
    /// Identities-only resume: only real raw data may rebuild; flags do not supply fits/scores.
    #[must_use]
    pub fn resume(previous: StageIdentities, resume: ResumeContext) -> Self {
        Self {
            previous,
            live: None,
            boundary: Boundary::FreshProcess(ResumeContext {
                supplied_data: resume.supplied_data,
                ..ResumeContext::default()
            }),
            input_rows: 0,
        }
    }
    /// Actual state boundary.
    #[must_use]
    pub const fn boundary(&self) -> Boundary {
        self.boundary
    }
    /// Last successful declared stage inputs.
    #[must_use]
    pub const fn identities(&self) -> &StageIdentities {
        &self.previous
    }
    /// Whether actual prepared fit/score state exists.
    #[must_use]
    pub const fn is_live(&self) -> bool {
        self.live.is_some()
    }
    /// Actual marginal scores including original row ids and executed folds.
    #[must_use]
    pub fn score_table(&self) -> Option<&ScoreTable> {
        self.live.as_ref().and_then(|live| live.prepared.score_table())
    }
    /// Exact retained final CATE model; absent for DML or a nonportable learner.
    #[must_use]
    pub fn fitted_effect(&self) -> Option<&FittedEffect> {
        self.live
            .as_ref()
            .and_then(|live| live.result.estimate.as_effect())
            .and_then(|effect| effect.fitted_effect.as_deref())
    }
    /// Feature names in the retained CATE model's exact role order.
    #[must_use]
    pub fn prediction_columns(&self) -> Option<Vec<String>> {
        let live = self.live.as_ref()?;
        let model = self.fitted_effect()?;
        let super::builder::DataInput::Tabular(data) = &live.prepared.study().data else {
            return None;
        };
        let schema = data.schema();
        model
            .features
            .iter()
            .map(|&id| {
                schema.variables().iter().find(|v| v.id.raw() == id).map(|v| v.name.to_string())
            })
            .collect()
    }
    /// Shared plan for the actual estimator and retained state.
    #[must_use]
    pub fn plan(&self, request: &DrRequest, ctx: &ExecutionContext) -> RecalcPlan {
        self.plan_ids(&request.identities(ctx), request)
    }
    fn plan_ids(&self, ids: &StageIdentities, request: &DrRequest) -> RecalcPlan {
        RecalcPlan::plan(
            &self.previous,
            ids,
            &RecalcCapabilities {
                retarget: RetargetSupport::Licensed,
                request: if request.estimator.supported() {
                    RequestSupport::OnGrid
                } else {
                    RequestSupport::Unsupported { licensed_route: None }
                },
                boundary: self.boundary,
            },
        )
    }
    /// Predict model-point CATE values. This supplies no profile uncertainty or positivity claim.
    /// # Errors
    /// Missing predictor, different feature schema, nonfinite/outside retained covariate ranges,
    /// cancellation/workspace or learner prediction failures.
    pub fn predict(
        &self,
        features: &[VariableId],
        columns: &[&[f64]],
        nrows: usize,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, RecalcRunError> {
        let live = self.live.as_ref().ok_or(RecalcRunError::NoLiveState(Stage::ScoreArtifact))?;
        let model = self
            .fitted_effect()
            .ok_or(RecalcRunError::Request("recalc.dr_predictor_unavailable"))?;
        if features.iter().map(|id| id.raw()).collect::<Vec<_>>() != model.features
            || columns.len() != features.len()
            || columns.iter().any(|col| col.len() != nrows || col.iter().any(|x| !x.is_finite()))
        {
            return Err(RecalcRunError::Request("recalc.dr_prediction_schema_mismatch"));
        }
        for (col, (_, lo, hi)) in columns.iter().zip(&live.bounds) {
            if col.iter().any(|x| x < lo || x > hi) {
                return Err(RecalcRunError::Request("recalc.dr_prediction_out_of_support"));
            }
        }
        model
            .predict(features, columns, nrows, ctx)
            .map_err(|e| RecalcRunError::Execution(e.into()))
    }
    /// Export frozen marginal scores for the existing independent score resume consumer.
    /// # Errors
    /// Missing live scores, or an invalid frozen score/identity binding.
    pub fn export_scores(&self) -> Result<FrozenScoreTable, RecalcRunError> {
        super::recalc_cell::freeze(
            self.score_table().ok_or(RecalcRunError::NoLiveState(Stage::ScoreArtifact))?,
            &self.previous,
            self.input_rows,
            RetargetSupport::Licensed,
        )
    }
    /// Export the original fitted response and predictor through the existing result codec.
    /// The producing RNG/context is retained: an export context does not redefine the fit.
    /// # Errors
    /// Missing predictor/live result, or the contracted-result codec refuses its source binding.
    pub fn export_predictor_result(
        &self,
        _ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, RecalcRunError> {
        let live = self.live.as_ref().ok_or(RecalcRunError::NoLiveState(Stage::ScoreArtifact))?;
        if self.fitted_effect().is_none() {
            return Err(RecalcRunError::Request("recalc.dr_predictor_unavailable"));
        }
        Ok(live.prepared.encode_contracted_result(
            &live.result,
            "recalc-dr-predictor",
            &live.fit_context,
        )?)
    }
}
fn recomputed(plan: &RecalcPlan, stage: Stage) -> bool {
    matches!(plan.status(stage), Some(StageStatus::Recomputed { .. }))
}
/// Execute checked preparation/refresh, score-law retarget and decision stages with actual fits.
/// # Errors
/// Unsupported PLR/trimmed requests, unavailable state, invalid roles/weights, failed checked
/// preparation/fits or receipts. Failures leave the prior successful state unchanged.
pub fn execute_dr_with_receipt(
    session: &mut DrSession,
    request: &DrRequest,
    ctx: &ExecutionContext,
) -> Result<RecalcOutcome, RecalcRunError> {
    request.validate()?;
    let requested = request.identities(ctx);
    let plan = session.plan_ids(&requested, request);
    if !plan.is_executable() {
        return Err(RecalcRunError::Refused(Box::new(plan)));
    }
    if ctx.cancellation.is_cancelled() {
        return Err(RecalcRunError::Request("recalc.cancelled"));
    }
    if let (Some(target), Some(table)) = (&request.target, session.score_table()) {
        if !recomputed(&plan, Stage::Identification)
            && !recomputed(&plan, Stage::ScoreArtifact)
            && (target.weights.len() != table.n_rows
                || target.depends_on.iter().any(|id| !table.adjustment_set.contains(id)))
        {
            return Err(RecalcRunError::Request("recalc.invalid_target_weights"));
        }
    }
    if recomputed(&plan, Stage::ScoreArtifact) {
        let n = request.columns.first().map_or(0, |(_, v)| v.len()) as u64;
        let p = request.columns.len() as u64;
        let bytes = n
            .saturating_mul(p.saturating_add(32))
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
    }
    let mut recorder = ReceiptRecorder::new();
    let held = session.live.as_ref();
    let (prepared, result, fit_context, bounds) = if recomputed(&plan, Stage::ScoreArtifact) {
        let cache = CrossfitNuisanceCache::new();
        let (((built, score_builds), final_fits), all_fits) = count_resolved_fits(|| {
            count_dr_final_fits(|| {
                count_aipw_score_builds(|| {
                    cache
                        .scope(|| -> Result<_, RecalcRunError> {
                            let data = request.data()?;
                            let (prepared, result) = if recomputed(&plan, Stage::Identification) {
                                let prepared = request.study(data)?.prepare(ctx)?;
                                recorder.record(Counter::Identification, 1);
                                let result = prepared.estimate_retained(ctx)?;
                                (prepared, result)
                            } else {
                                let mut prepared = held
                                    .ok_or(RecalcRunError::NoLiveState(Stage::Identification))?
                                    .prepared
                                    .as_ref()
                                    .clone();
                                let result = prepared.refresh_dr_configuration(
                                    data,
                                    request.estimator.spec(),
                                    ctx,
                                )?;
                                (prepared, result)
                            };
                            if prepared.score_table().is_none() {
                                return Err(RecalcRunError::Request(
                                    "recalc.dr_scores_unavailable",
                                ));
                            }
                            let mut bounds = Vec::new();
                            if let Some(model) =
                                result.estimate.as_effect().and_then(|e| e.fitted_effect.as_ref())
                            {
                                let table = prepared.score_table().ok_or(
                                    RecalcRunError::Request("recalc.dr_scores_unavailable"),
                                )?;
                                for &id in &model.features {
                                    let values = &request.columns[id as usize].1;
                                    let lo = table
                                        .row_index
                                        .iter()
                                        .map(|&row| values[row as usize])
                                        .fold(f64::INFINITY, f64::min);
                                    let hi = table
                                        .row_index
                                        .iter()
                                        .map(|&row| values[row as usize])
                                        .fold(f64::NEG_INFINITY, f64::max);
                                    bounds.push((VariableId::from_raw(id), lo, hi));
                                }
                            }
                            Ok((Arc::new(prepared), Arc::new(result), ctx.clone(), bounds))
                        })
                        .0
                })
            })
        });
        recorder.record(
            Counter::FoldFit,
            all_fits
                .checked_sub(final_fits)
                .ok_or(RecalcRunError::Request("recalc.dr_fit_counter_mismatch"))?,
        );
        recorder.record(Counter::ModelFit, final_fits);
        recorder.record(Counter::ScoreComputation, score_builds);
        built?
    } else {
        let live = held.ok_or(RecalcRunError::NoLiveState(Stage::ScoreArtifact))?;
        (
            Arc::clone(&live.prepared),
            Arc::clone(&live.result),
            live.fit_context.clone(),
            live.bounds.clone(),
        )
    };
    let law = if recomputed(&plan, Stage::Law) {
        let table = prepared
            .score_table()
            .ok_or(RecalcRunError::Request("recalc.dr_scores_unavailable"))?;
        let ones;
        let (weights, parents) = if let Some(target) = &request.target {
            (target.weights.as_slice(), target.depends_on.as_slice())
        } else {
            ones = vec![1.; table.n_rows];
            (ones.as_slice(), &[][..])
        };
        let retarget = prepared.retarget(weights, parents, ctx)?;
        let effect = retarget
            .estimate
            .as_effect()
            .ok_or(RecalcRunError::Request("recalc.dr_effect_unavailable"))?;
        recorder.record(Counter::Reweight, 1);
        LawValue { ate: effect.ate, std_error: effect.se_analytic }
    } else {
        held.ok_or(RecalcRunError::NoLiveState(Stage::Law))?.law
    };
    let decision = if recomputed(&plan, Stage::Decision) {
        decide(law, request.utility, &mut recorder)?
    } else {
        held.ok_or(RecalcRunError::NoLiveState(Stage::Decision))?.decision
    };
    let receipt = recorder.finish(&plan)?;
    session.live = Some(Live { prepared, result, fit_context, law, decision, bounds });
    session.previous = requested;
    session.boundary = Boundary::InProcess;
    session.input_rows = request.columns.first().map_or(0, |(_, v)| v.len()) as u64;
    Ok(RecalcOutcome { plan, receipt, law, decision })
}
