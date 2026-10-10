//! Retained, native-issued Bayesian posterior projections under the shared planner.
//! Calibrated joint-transport publication remains a separate closed route.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use super::recalc_receipt::{DecisionValue, RecalcReceipt, RecalcRunError, UtilitySpec};
use crate::BayesianConfig;
use antecedent_core::{VariableId, recalc::RecalcPlan};
use antecedent_data::TableView;
use antecedent_estimate::CausalPosterior;

/// Existing checked Gaussian-identity model family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BayesianModel {
    /// Existing conjugate linear g-computation model.
    Gaussian,
    /// Existing checked quadratic-basis mean-effect model.
    QuadraticBasis,
}
/// Requested summaries of the effect draws issued by native checked execution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PosteriorSummarySpec {
    /// Lower posterior probability, strictly inside `(0,1)`.
    pub lower_probability: f64,
    /// Upper posterior probability, above the lower probability.
    pub upper_probability: f64,
    /// Effect threshold for the posterior probability of a smaller effect.
    pub threshold: f64,
}
impl Default for PosteriorSummarySpec {
    fn default() -> Self {
        Self { lower_probability: 0.025, upper_probability: 0.975, threshold: 0.0 }
    }
}
impl PosteriorSummarySpec {
    pub(super) fn validate(self) -> Result<(), RecalcRunError> {
        if !self.lower_probability.is_finite()
            || !self.upper_probability.is_finite()
            || !self.threshold.is_finite()
            || self.lower_probability <= 0.0
            || self.upper_probability >= 1.0
            || self.lower_probability >= self.upper_probability
        {
            return Err(RecalcRunError::Request("recalc.bayesian_invalid_summary"));
        }
        Ok(())
    }
}
/// Raw checked-study request. Posterior draws are never accepted as caller input.
#[derive(Clone, Debug)]
pub struct BayesianRequest {
    /// Graph-order numeric columns, with missing cells retained for checked complete cases.
    pub columns: Vec<(String, Vec<f64>)>,
    /// Directed edges in that stable graph order.
    pub edges: Vec<(u32, u32)>,
    /// Binary intervention role.
    pub treatment: VariableId,
    /// Continuous outcome role.
    pub outcome: VariableId,
    /// Existing engine family.
    pub model: BayesianModel,
    /// Existing inference and checked mapped prior-transfer configuration.
    pub inference: BayesianConfig,
    /// Draw projection; changing it does not fit or sample a model.
    pub summary: PosteriorSummarySpec,
    /// Expected-net-benefit rule.
    pub utility: UtilitySpec,
}
/// Empirical projections of a retained native posterior, with posterior SD rather than SE.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BayesianLaw {
    /// Posterior effect mean.
    pub mean: f64,
    /// Posterior effect standard deviation.
    pub standard_deviation: f64,
    /// Empirical posterior quantile at the requested lower probability.
    pub lower_quantile: f64,
    /// Empirical posterior quantile at the requested upper probability.
    pub upper_quantile: f64,
    /// Posterior proportion strictly below the requested effect threshold.
    pub probability_below: f64,
    /// Actual retained aligned rows.
    pub draws: usize,
}
/// Checked posterior projection and actual-work receipt.
#[derive(Debug)]
pub struct BayesianRecalcOutcome {
    /// Shared dependency plan.
    pub plan: RecalcPlan,
    /// Actual model solves, emitted posterior rows and projection work.
    pub receipt: RecalcReceipt,
    /// Native-issued effect posterior projection.
    pub law: BayesianLaw,
    /// Decision from the posterior expected effect.
    pub decision: DecisionValue,
}

// This helper accepts only the native posterior held by the session. Public
// bindings do not grant its authority to caller-created projections or arrays.
fn summarize_native_posterior(
    posterior: &CausalPosterior,
    spec: PosteriorSummarySpec,
) -> Result<BayesianLaw, RecalcRunError> {
    spec.validate()?;
    let column = posterior
        .effect_column()
        .ok_or(RecalcRunError::Request("recalc.bayesian_effect_unavailable"))?;
    let values = posterior
        .draws
        .column(column)
        .map_err(|_| RecalcRunError::Request("recalc.bayesian_effect_unavailable"))?;
    summarize_issued_values(values, spec)
}
#[allow(clippy::cast_precision_loss, reason = "bounded native posterior row counts")]
pub(super) fn summarize_issued_values(
    values: &[f64],
    spec: PosteriorSummarySpec,
) -> Result<BayesianLaw, RecalcRunError> {
    antecedent_core::execution_attempt::run_operation(
        antecedent_core::execution_attempt::Operation::LawSummary,
        || {
            if values.len() < 2 || values.iter().any(|v| !v.is_finite()) {
                return Err(RecalcRunError::Request("recalc.bayesian_effect_unavailable"));
            }
            let n = values.len() as f64;
            let anchor = values[0];
            let mean = anchor + values.iter().map(|v| v - anchor).sum::<f64>() / n;
            let standard_deviation =
                (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
            if !mean.is_finite() || !standard_deviation.is_finite() {
                return Err(RecalcRunError::Request("recalc.bayesian_effect_unavailable"));
            }
            let probability_below =
                values.iter().filter(|&&v| v < spec.threshold).count() as f64 / n;
            let mut ordered = values.to_vec();
            ordered.sort_by(f64::total_cmp);
            Ok(BayesianLaw {
                mean,
                standard_deviation,
                lower_quantile: empirical_quantile(&ordered, spec.lower_probability),
                upper_quantile: empirical_quantile(&ordered, spec.upper_probability),
                probability_below,
                draws: values.len(),
            })
        },
    )
}
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "validated probability inside (0,1) selects a bounded native draw index"
)]
fn empirical_quantile(ordered: &[f64], probability: f64) -> f64 {
    let position = probability * (ordered.len() - 1) as f64;
    let index = position.floor() as usize;
    let fraction = position - index as f64;
    ordered[index] + fraction * (ordered[(index + 1).min(ordered.len() - 1)] - ordered[index])
}

impl BayesianRequest {
    fn data(&self) -> Result<antecedent_data::TabularData, RecalcRunError> {
        antecedent_data::TabularData::from_f64_columns(
            self.columns
                .iter()
                .map(|(name, values)| (name.as_str(), values.as_slice()))
                .collect::<Vec<_>>(),
        )
        .map_err(|_| RecalcRunError::Request("recalc.bayesian_invalid_data"))
    }
    fn graph(&self) -> Result<antecedent_graph::Dag, RecalcRunError> {
        let n = u32::try_from(self.columns.len())
            .map_err(|_| RecalcRunError::Request("recalc.bayesian_invalid_graph"))?;
        let mut graph = antecedent_graph::Dag::with_variables(n);
        let mut edges = self.edges.clone();
        edges.sort_unstable();
        edges.dedup();
        for (from, to) in edges {
            graph
                .insert_directed(
                    antecedent_graph::DenseNodeId::from_raw(from),
                    antecedent_graph::DenseNodeId::from_raw(to),
                )
                .map_err(|_| RecalcRunError::Request("recalc.bayesian_invalid_graph"))?;
        }
        Ok(graph)
    }
    fn study(&self, data: antecedent_data::TabularData) -> Result<crate::Study, RecalcRunError> {
        let builder = crate::Study::tabular(data)
            .graph(self.graph()?)
            .query(antecedent_core::AverageEffectQuery::binary_ate(self.treatment, self.outcome))
            .inference(crate::InferenceMode::Bayesian(self.inference.clone()))
            .refute(crate::RefuteSuite::None);
        let builder = match self.model {
            BayesianModel::Gaussian => builder,
            BayesianModel::QuadraticBasis => {
                builder.estimator(crate::strategy_table::EstimatorId::BayesianBasisGcomp)
            }
        };
        Ok(builder.build()?)
    }
    fn validate(&self) -> Result<(), RecalcRunError> {
        if self.inference.external_compose.is_some() {
            return Err(RecalcRunError::Request("recalc.bayesian_external_compose_unsupported"));
        }
        if let Some(bytes) = &self.inference.prior_artifact {
            if !bytes.starts_with(antecedent_io::bayesian_prior_recalc_artifact::BOUND_PRIOR_MAGIC)
            {
                return Err(RecalcRunError::Request("recalc.bayesian_prior_source_unverified"));
            }
            if self.inference.prior.is_some() {
                return Err(RecalcRunError::Request("recalc.bayesian_prior_source_conflict"));
            }
        }

        if let Some(prior) = &self.inference.prior {
            source_prior_wire(prior)?;
            if prior.specs.iter().any(|spec| spec.validate().is_err()) {
                return Err(RecalcRunError::Request("recalc.bayesian_prior_invalid"));
            }
        }
        self.summary.validate()?;
        self.utility.validate()?;
        let rows = self.columns.first().map_or(0, |(_, values)| values.len());
        let names =
            self.columns.iter().map(|(name, _)| name).collect::<std::collections::BTreeSet<_>>();
        if rows == 0
            || rows > 100_000
            || self.columns.is_empty()
            || self.columns.len() > 128
            || rows.saturating_mul(self.columns.len()) > 1_000_000
            || names.len() != self.columns.len()
            || self.columns.iter().any(|(name, values)| {
                name.is_empty()
                    || values.len() != rows
                    || values.iter().any(|value| value.is_infinite())
            })
        {
            return Err(RecalcRunError::Request("recalc.bayesian_invalid_data"));
        }
        if self.treatment == self.outcome
            || self.treatment.raw() as usize >= self.columns.len()
            || self.outcome.raw() as usize >= self.columns.len()
        {
            return Err(RecalcRunError::Request("recalc.bayesian_invalid_query"));
        }
        if self.inference.backend != antecedent_estimate::BayesianBackendKind::ConjugateGaussian
            || self.inference.likelihood != antecedent_prob::BayesLikelihood::GaussianIdentity
            || !(2..=100_000).contains(&self.inference.n_draws)
            || !self.inference.prior_scale.is_finite()
            || self.inference.prior_scale <= 0.0
            || self.inference.prior_artifact.as_ref().is_some_and(|bytes| bytes.len() > 16_000_000)
        {
            return Err(RecalcRunError::Request("recalc.bayesian_inference_unsupported"));
        }
        if self.model == BayesianModel::QuadraticBasis
            && (self.inference.prior.is_some()
                || self.inference.prior_artifact.is_some()
                || self.inference.external_compose.is_some())
        {
            return Err(RecalcRunError::Request("recalc.bayesian_basis_prior_unsupported"));
        }
        self.graph()?;
        Ok(())
    }
}

struct BayesianLive {
    source_edges: Vec<(u32, u32)>,
    model: BayesianModel,
    verified_prior: Option<std::sync::Arc<VerifiedBayesianPrior>>,
    prepared: std::sync::Arc<crate::PreparedStudy>,
    result: std::sync::Arc<crate::StudyResult>,
    law: BayesianLaw,
    decision: DecisionValue,
    context: antecedent_core::ExecutionContext,
}
/// Retains only posteriors issued by the original checked native study engine.
pub struct BayesianSession {
    previous: antecedent_core::recalc::StageIdentities,
    live: Option<BayesianLive>,
    boundary: antecedent_core::recalc::Boundary,
}
impl Default for BayesianSession {
    fn default() -> Self {
        Self::new()
    }
}
impl BayesianSession {
    /// Empty in-process native session.
    #[must_use]
    pub fn new() -> Self {
        Self {
            previous: antecedent_core::recalc::StageIdentities::new(),
            live: None,
            boundary: antecedent_core::recalc::Boundary::InProcess,
        }
    }
    /// Resume semantic identities. Caller flags cannot create a retained posterior.
    #[must_use]
    pub fn resume(
        previous: antecedent_core::recalc::StageIdentities,
        context: antecedent_core::recalc::ResumeContext,
    ) -> Self {
        Self {
            previous,
            live: None,
            boundary: antecedent_core::recalc::Boundary::FreshProcess(
                antecedent_core::recalc::ResumeContext {
                    supplied_data: context.supplied_data,
                    ..antecedent_core::recalc::ResumeContext::default()
                },
            ),
        }
    }
    /// Last successful semantic identities.
    #[must_use]
    pub const fn identities(&self) -> &antecedent_core::recalc::StageIdentities {
        &self.previous
    }
    /// Actual native state boundary.
    #[must_use]
    pub const fn boundary(&self) -> antecedent_core::recalc::Boundary {
        self.boundary
    }
    /// Actual retained checked state, never an optimistic supplied flag.
    #[must_use]
    pub const fn is_live(&self) -> bool {
        self.live.is_some()
    }
    /// Borrow the original native-issued scientific result.
    #[must_use]
    pub fn result(&self) -> Option<&crate::StudyResult> {
        self.live.as_ref().map(|live| live.result.as_ref())
    }
    /// Exact retained producing schema for opaque issued result authority.
    #[must_use]
    pub fn schema(&self) -> Option<&antecedent_core::CausalSchema> {
        let live = self.live.as_ref()?;
        let super::builder::DataInput::Tabular(data) = &live.prepared.study().data else {
            return None;
        };
        Some(data.schema())
    }
    /// Actual producing execution context of the held scientific posterior.
    #[must_use]
    pub fn producing_context(&self) -> Option<&antecedent_core::ExecutionContext> {
        self.live.as_ref().map(|live| &live.context)
    }
    /// Borrow the original aligned causal posterior, without accepting caller-created draws.
    #[must_use]
    pub fn posterior(&self) -> Option<&antecedent_estimate::CausalPosterior> {
        self.result().and_then(|result| result.posterior.as_ref())
    }
    /// Export the original contracted result with its actual producing seed and configuration.
    /// # Errors
    /// No live posterior, cancelled export, or invalid result contract.
    pub fn export_result(
        &self,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Vec<u8>, RecalcRunError> {
        if ctx.cancellation.is_cancelled() {
            return Err(RecalcRunError::Request("recalc.cancelled"));
        }
        let live = self
            .live
            .as_ref()
            .ok_or(RecalcRunError::NoLiveState(antecedent_core::recalc::Stage::ScoreArtifact))?;
        Ok(live.prepared.encode_contracted_result(
            &live.result,
            "recalc-bayesian",
            &live.context,
        )?)
    }
}

fn bayesian_literal(label: &str, text: &str) -> antecedent_core::recalc::StageIdentity {
    antecedent_core::recalc::StageIdentity::of(label, &[text.as_bytes()])
}
impl BayesianRequest {
    /// Semantic identity of the causal claim, actual input, posterior and draw projection.
    #[must_use]
    pub fn identities(
        &self,
        ctx: &antecedent_core::ExecutionContext,
    ) -> antecedent_core::recalc::StageIdentities {
        use antecedent_core::recalc::{Stage, StageIdentities, StageIdentity};
        let mut ids = StageIdentities::new();
        let graph = super::recalc_cell::graph_identity(self.columns.len(), &self.edges);
        let names = self.columns.iter().map(|(name, _)| name.as_bytes()).collect::<Vec<_>>();
        let mut graph_parts = vec![graph.as_bytes().as_slice()];
        graph_parts.extend(names);
        ids.set(Stage::Graph, StageIdentity::of("bayesian.graph_schema.v1", &graph_parts));
        ids.set(
            Stage::Query,
            bayesian_literal(
                "bayesian.query.v1",
                &format!(
                    "binary.mean:{}:{}:{:?}",
                    self.treatment.raw(),
                    self.outcome.raw(),
                    self.model,
                ),
            ),
        );
        ids.set(Stage::Regime, bayesian_literal("bayesian.regime.v1", "static.binary.0_1"));
        ids.set(
            Stage::Evidence,
            StageIdentity::of(
                "bayesian.evidence.v2",
                &[
                    b"checked.observational",
                    self.inference.prior_artifact.as_deref().unwrap_or(&[]),
                ],
            ),
        );
        ids.set(Stage::SourcePopulation, bayesian_literal("bayesian.source.v1", "all_observed"));
        ids.set(Stage::TargetPopulation, bayesian_literal("bayesian.target.v1", "all_observed"));
        ids.set(Stage::RowDesign, bayesian_literal("bayesian.rows.v1", "checked.complete_cases"));
        ids.set(Stage::TreatmentGrid, bayesian_literal("bayesian.grid.v1", "binary.0_1"));
        let mut raw = Vec::new();
        for (name, values) in &self.columns {
            raw.push(name.as_bytes().to_vec());
            raw.push(values.iter().flat_map(|value| value.to_bits().to_le_bytes()).collect());
        }
        ids.set(
            Stage::DataSnapshot,
            StageIdentity::of(
                "bayesian.snapshot.v1",
                &raw.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            ),
        );
        // Prior artifact bytes are hashed directly instead of Debug-formatting
        // an arbitrarily large hexadecimal representation.
        let config = format!(
            "{:?}:{:?}:{}:{}:{}:{:?}:{:?}:{:?}:{}:{:?}:{:?}",
            self.inference.backend,
            self.inference.likelihood,
            self.inference.n_draws,
            self.inference.n_draws_explicit,
            self.inference.prior_scale.to_bits(),
            self.inference.prior,
            self.inference.prior_mapping,
            self.inference.external_compose,
            self.inference.prior_artifact.is_some(),
            ctx.rng.master_seed(),
            ctx.adaptive_draws,
        );
        ids.set(
            Stage::LearnerFoldsRng,
            StageIdentity::of(
                "bayesian.inference.v1",
                &[config.as_bytes(), self.inference.prior_artifact.as_deref().unwrap_or(&[])],
            ),
        );
        ids.set(Stage::Identification, bayesian_literal("bayesian.proof.v1", "checked.backdoor"));
        ids.set(
            Stage::ScoreArtifact,
            bayesian_literal("bayesian.posterior.v1", "native.issued.aligned"),
        );
        ids.set(
            Stage::Law,
            bayesian_literal("bayesian.summary.v1", &format!("{:?}", self.summary)),
        );
        ids.set(Stage::Utility, super::recalc_cell::utility_identity(self.utility));
        ids.set(
            Stage::Decision,
            bayesian_literal("bayesian.decision.v1", "net_benefit.posterior_mean"),
        );
        ids
    }
}

impl BayesianRequest {
    fn fit_workspace_guard(
        &self,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<(), RecalcRunError> {
        const MAX_WORKSPACE_VALUES: usize = 16_000_000;
        if ctx.cancellation.is_cancelled() {
            return Err(RecalcRunError::Request("recalc.cancelled"));
        }
        let rows = self.columns.first().map_or(0, |(_, values)| values.len());
        // All columns are an upper bound on checked adjustment coordinates.
        let p = match self.model {
            BayesianModel::Gaussian => self.columns.len().saturating_add(2),
            BayesianModel::QuadraticBasis => self.columns.len().saturating_mul(3).saturating_add(2),
        };
        let values = rows
            .saturating_mul(p)
            .saturating_mul(4)
            .saturating_add(p.saturating_mul(p).saturating_mul(12))
            .saturating_add(
                self.inference.n_draws.saturating_mul(p.saturating_add(4)).saturating_mul(2),
            )
            .saturating_add(if self.model == BayesianModel::QuadraticBasis {
                rows.saturating_mul(self.inference.n_draws).saturating_mul(3)
            } else {
                0
            });
        if values > MAX_WORKSPACE_VALUES {
            return Err(RecalcRunError::Request("recalc.bayesian_workspace_limit"));
        }
        let bytes = u64::try_from(values).unwrap_or(u64::MAX).saturating_mul(16);
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
}

fn bayesian_plan(
    previous: &antecedent_core::recalc::StageIdentities,
    requested: &antecedent_core::recalc::StageIdentities,
    boundary: antecedent_core::recalc::Boundary,
    supported: bool,
) -> RecalcPlan {
    RecalcPlan::plan(
        previous,
        requested,
        &antecedent_core::recalc::RecalcCapabilities {
            retarget: antecedent_core::recalc::RetargetSupport::NotDeclared,
            request: if supported {
                antecedent_core::recalc::RequestSupport::OnGrid
            } else {
                antecedent_core::recalc::RequestSupport::Unsupported { licensed_route: None }
            },
            boundary,
        },
    )
}
fn recomputed(plan: &RecalcPlan, stage: antecedent_core::recalc::Stage) -> bool {
    matches!(plan.status(stage), Some(antecedent_core::recalc::StageStatus::Recomputed { .. }))
}
impl BayesianSession {
    /// Plan dependencies over native retained state and actual raw request identities.
    #[must_use]
    pub fn plan(
        &self,
        request: &BayesianRequest,
        ctx: &antecedent_core::ExecutionContext,
    ) -> RecalcPlan {
        bayesian_plan(
            &self.previous,
            &request.identities(ctx),
            self.boundary,
            request.validate().is_ok(),
        )
    }
}
/// Execute the original checked Bayesian study engine or project its retained native draws.
/// # Errors
/// Unsupported inference, invalid data/graph/prior, ownership double use, budgets, or receipts.
pub fn execute_bayesian_with_receipt(
    session: &mut BayesianSession,
    request: &BayesianRequest,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<BayesianRecalcOutcome, RecalcRunError> {
    use super::recalc_receipt::{Counter, LawValue, ReceiptRecorder, decide};
    use antecedent_core::recalc::{Boundary, Stage};
    use antecedent_expr::execution_counts::{StaticWork, count_static_work, note_static_work};
    use antecedent_identify::execution_counts::count_checked_identifications;
    use antecedent_prob::fit_counts::count_bayesian_work;
    request.validate()?;
    let identities = request.identities(ctx);
    let plan = bayesian_plan(&session.previous, &identities, session.boundary, true);
    if !plan.is_executable() {
        return Err(RecalcRunError::Refused(Box::new(plan)));
    }
    if ctx.cancellation.is_cancelled() {
        return Err(RecalcRunError::Request("recalc.cancelled"));
    }
    let old = session.live.as_ref();
    let mut recorder = ReceiptRecorder::new();
    let (prepared, result, context, verified_prior) = if recomputed(&plan, Stage::ScoreArtifact) {
        request.fit_workspace_guard(ctx)?;
        let ((execution, checks), work) = count_bayesian_work(|| {
            count_checked_identifications(|| -> Result<_, RecalcRunError> {
                let data = request.data()?;
                let verified_prior = if let Some(bytes) = &request.inference.prior_artifact {
                    let prior = if let Some(cached) = old
                        .and_then(|live| live.verified_prior.as_ref())
                        .filter(|prior| prior.envelope.as_ref() == bytes.as_ref())
                    {
                        if cached.source_snapshot == data.storage().content_digest() {
                            return Err(RecalcRunError::Request(
                                "recalc.bayesian_prior_likelihood_double_use",
                            ));
                        }
                        std::sync::Arc::clone(cached)
                    } else {
                        std::sync::Arc::new(verify_bound_bayesian_prior(bytes, &data, ctx)?)
                    };
                    Some(prior)
                } else {
                    None
                };
                let mut execution_request = request.clone();
                execution_request.inference.prior_artifact =
                    verified_prior.as_ref().map(|prior| std::sync::Arc::clone(&prior.posterior));
                if recomputed(&plan, Stage::Identification) {
                    let prepared = execution_request.study(data)?.prepare(ctx)?;
                    let result = prepared.estimate_retained(ctx)?;
                    Ok((prepared, result, verified_prior))
                } else {
                    let retained = old.ok_or(RecalcRunError::NoLiveState(Stage::Identification))?;
                    let mut prepared = retained
                        .prepared
                        .rebind_checked_bayesian_inference(execution_request.inference.clone())?;
                    let result = prepared.refresh(data, ctx)?;
                    Ok((prepared, result, verified_prior))
                }
            })
        });
        recorder.record(Counter::Identification, checks);
        recorder.record(Counter::ModelFit, work.model_fits);
        recorder.record(Counter::PosteriorDraw, work.posterior_draws);
        let (prepared, result, verified_prior) = execution?;
        (std::sync::Arc::new(prepared), std::sync::Arc::new(result), ctx.clone(), verified_prior)
    } else {
        let retained = old.ok_or(RecalcRunError::NoLiveState(Stage::ScoreArtifact))?;
        (
            std::sync::Arc::clone(&retained.prepared),
            std::sync::Arc::clone(&retained.result),
            retained.context.clone(),
            retained.verified_prior.as_ref().map(std::sync::Arc::clone),
        )
    };
    let law = if recomputed(&plan, Stage::Law) {
        let (law, work) = count_static_work(|| {
            let posterior = result
                .posterior
                .as_ref()
                .ok_or(RecalcRunError::Request("recalc.bayesian_effect_unavailable"))?;
            let law = summarize_native_posterior(posterior, request.summary)?;
            note_static_work(StaticWork::LawSummary);
            Ok::<_, RecalcRunError>(law)
        });
        recorder.record(Counter::LawSummary, work.law_summaries);
        law?
    } else {
        old.ok_or(RecalcRunError::NoLiveState(Stage::Law))?.law
    };
    let decision = if recomputed(&plan, Stage::Decision) {
        decide(LawValue { ate: law.mean, std_error: f64::NAN }, request.utility, &mut recorder)?
    } else {
        old.ok_or(RecalcRunError::NoLiveState(Stage::Decision))?.decision
    };
    let receipt = recorder.finish(&plan)?;
    session.live = Some(BayesianLive {
        prepared,
        result,
        law,
        decision,
        context,
        source_edges: request.edges.clone(),
        model: request.model,
        verified_prior,
    });
    session.previous = identities;
    session.boundary = Boundary::InProcess;
    Ok(BayesianRecalcOutcome { plan, receipt, law, decision })
}

struct VerifiedBayesianPrior {
    envelope: std::sync::Arc<[u8]>,
    source_snapshot: [u8; 32],
    posterior: std::sync::Arc<[u8]>,
}
fn source_prior_wire(
    prior: &antecedent_prob::PriorSet,
) -> Result<Vec<antecedent_io::bayesian_prior_recalc_artifact::BoundPriorSpecWire>, RecalcRunError>
{
    use antecedent_io::bayesian_prior_recalc_artifact::BoundPriorSpecWire as Wire;
    if prior.specs.len() > 16
        || !prior.restrictions.is_empty()
        || prior.contrast.is_some()
        || !prior.categorical.is_empty()
    {
        return Err(RecalcRunError::Request("recalc.bayesian_prior_source_unsupported"));
    }
    prior
        .specs
        .iter()
        .map(|spec| {
            Ok(match spec {
                antecedent_prob::PriorSpec::GaussianCoefficients(p) => {
                    if p.mean.len() > 128 || p.variance.len() > 128 {
                        return Err(RecalcRunError::Request(
                            "recalc.bayesian_prior_source_unsupported",
                        ));
                    }
                    Wire::Gaussian { mean: p.mean.to_vec(), variance: p.variance.to_vec() }
                }
                antecedent_prob::PriorSpec::KnownResidualVariance(v) => Wire::KnownVariance(*v),
                antecedent_prob::PriorSpec::ResidualInvGamma(p) => {
                    Wire::InvGamma { shape: p.shape, scale: p.scale }
                }
                antecedent_prob::PriorSpec::CoefficientCorrelation(c) => {
                    if c.dim() > 128 || c.matrix().len() > 16_384 {
                        return Err(RecalcRunError::Request(
                            "recalc.bayesian_prior_source_unsupported",
                        ));
                    }
                    Wire::Correlation { dim: c.dim(), matrix: c.matrix().to_vec() }
                }
            })
        })
        .collect()
}
fn source_prior_from_wire(
    specs: &[antecedent_io::bayesian_prior_recalc_artifact::BoundPriorSpecWire],
) -> Result<antecedent_prob::PriorSet, RecalcRunError> {
    use antecedent_io::bayesian_prior_recalc_artifact::BoundPriorSpecWire as Wire;
    if specs.len() > 16 {
        return Err(RecalcRunError::Request("recalc.bayesian_prior_source_unverified"));
    }
    let mut prior = antecedent_prob::PriorSet::new();
    for spec in specs {
        prior.push(match spec {
            Wire::Gaussian { mean, variance } if mean.len() <= 128 && variance.len() <= 128 => {
                antecedent_prob::PriorSpec::GaussianCoefficients(
                    antecedent_prob::GaussianCoefficientPrior {
                        mean: mean.clone().into(),
                        variance: variance.clone().into(),
                    },
                )
            }
            Wire::KnownVariance(v) => antecedent_prob::PriorSpec::KnownResidualVariance(*v),
            Wire::InvGamma { shape, scale } => {
                antecedent_prob::PriorSpec::ResidualInvGamma(antecedent_prob::InvGammaPrior {
                    shape: *shape,
                    scale: *scale,
                })
            }
            Wire::Correlation { dim, matrix } if *dim <= 128 && matrix.len() <= 16_384 => {
                antecedent_prob::PriorSpec::CoefficientCorrelation(
                    antecedent_prob::CoefficientCorrelation::new(*dim, matrix.clone()).map_err(
                        |_| RecalcRunError::Request("recalc.bayesian_prior_source_unverified"),
                    )?,
                )
            }
            _ => return Err(RecalcRunError::Request("recalc.bayesian_prior_source_unverified")),
        });
    }
    Ok(prior)
}
impl BayesianSession {
    /// Export an independently replayable source-data-bound native prior envelope.
    /// Only original Gaussian fits with bounded declared priors can be source artifacts;
    /// prior-transfer chains and adaptive sample schedules remain unavailable.
    /// # Errors
    /// No native state, unsupported source, or refused original posterior encoding.
    pub fn export_prior_source(&self) -> Result<Vec<u8>, RecalcRunError> {
        use antecedent_io::bayesian_prior_recalc_artifact::BoundBayesianPriorWire;
        let live = self
            .live
            .as_ref()
            .ok_or(RecalcRunError::NoLiveState(antecedent_core::recalc::Stage::ScoreArtifact))?;
        let crate::InferenceMode::Bayesian(config) = &live.prepared.study().inference else {
            return Err(RecalcRunError::Request("recalc.bayesian_prior_source_unsupported"));
        };
        if live.model != BayesianModel::Gaussian
            || config.prior_artifact.is_some()
            || config.external_compose.is_some()
            || live.context.adaptive_draws != antecedent_core::AdaptiveDrawBudget::disabled()
        {
            return Err(RecalcRunError::Request("recalc.bayesian_prior_source_unsupported"));
        }
        let super::builder::DataInput::Tabular(data) = &live.prepared.study().data else {
            return Err(RecalcRunError::Request("recalc.bayesian_prior_source_unsupported"));
        };
        let antecedent_core::CausalQuery::AverageEffect(query) = live.prepared.query() else {
            return Err(RecalcRunError::Request("recalc.bayesian_prior_source_unsupported"));
        };
        let columns = data
            .schema()
            .variables()
            .iter()
            .map(|variable| {
                data.float64_values(variable.id).map(|values| {
                    (
                        variable.name.to_string(),
                        values.iter().map(|value| value.to_bits()).collect(),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(crate::CausalError::from)?;
        let posterior = antecedent_io::encode_causal_posterior_bytes(
            self.posterior()
                .ok_or(RecalcRunError::Request("recalc.bayesian_effect_unavailable"))?,
            "recalc-bound-prior-source",
        )
        .map_err(crate::CausalError::Serialization)?;
        BoundBayesianPriorWire {
            columns,
            edges: live.source_edges.clone(),
            treatment: query.treatment.raw(),
            outcome: query.outcome.raw(),
            draws: config.n_draws,
            draws_explicit: config.n_draws_explicit,
            seed: live.context.rng.master_seed(),
            prior_scale: config.prior_scale,
            prior: config.prior.as_ref().map(source_prior_wire).transpose()?,
            source_snapshot: data.storage().content_digest(),
            posterior,
        }
        .export()
        .map_err(|e| crate::CausalError::Serialization(e).into())
    }
}
fn verify_bound_bayesian_prior(
    bytes: &[u8],
    target: &antecedent_data::TabularData,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<VerifiedBayesianPrior, RecalcRunError> {
    use antecedent_io::bayesian_prior_recalc_artifact::BoundBayesianPriorWire;
    let wire = BoundBayesianPriorWire::decode(bytes)
        .map_err(|_| RecalcRunError::Request("recalc.bayesian_prior_source_unverified"))?;
    let mut config = BayesianConfig::conjugate();
    config.n_draws = wire.draws;
    config.n_draws_explicit = wire.draws_explicit;
    config.prior_scale = wire.prior_scale;
    config.prior = wire.prior.as_ref().map(|prior| source_prior_from_wire(prior)).transpose()?;
    let request = BayesianRequest {
        columns: wire
            .columns
            .into_iter()
            .map(|(name, values)| (name, values.into_iter().map(f64::from_bits).collect()))
            .collect(),
        edges: wire.edges,
        treatment: VariableId::from_raw(wire.treatment),
        outcome: VariableId::from_raw(wire.outcome),
        model: BayesianModel::Gaussian,
        inference: config,
        summary: PosteriorSummarySpec::default(),
        utility: UtilitySpec { benefit_per_unit: 1.0, cost: 0.0 },
    };
    request.validate()?;
    let mut source_context = ctx.clone();
    source_context.rng = antecedent_core::RngFactory::from_seed(wire.seed);
    source_context.adaptive_draws = antecedent_core::AdaptiveDrawBudget::disabled();
    request.fit_workspace_guard(&source_context)?;
    let data = request.data()?;
    let source_snapshot = data.storage().content_digest();
    if source_snapshot != wire.source_snapshot {
        return Err(RecalcRunError::Request("recalc.bayesian_prior_source_unverified"));
    }
    if source_snapshot == target.storage().content_digest() {
        return Err(RecalcRunError::Request("recalc.bayesian_prior_likelihood_double_use"));
    }
    let result =
        request.study(data)?.prepare(&source_context)?.estimate_retained(&source_context)?;
    let posterior = result
        .posterior
        .as_ref()
        .ok_or(RecalcRunError::Request("recalc.bayesian_effect_unavailable"))?;
    let reproduced =
        antecedent_io::encode_causal_posterior_bytes(posterior, "recalc-bound-prior-source")
            .map_err(crate::CausalError::Serialization)?;
    if reproduced != wire.posterior {
        return Err(RecalcRunError::Request("recalc.bayesian_prior_source_unverified"));
    }
    let artifact = antecedent_io::read_and_migrate(reproduced.as_slice())
        .map_err(crate::CausalError::Serialization)?;
    let mut meta = antecedent_io::PriorSourceMeta::new(
        "recalc-bound-prior-source",
        antecedent_io::EstimandFingerprint::new(
            "average_effect",
            &request.columns[wire.treatment as usize].0,
            &request.columns[wire.outcome as usize].0,
        ),
        "nonparametrically_identified",
    );
    meta.provenance.insert(
        "bayesian.verified_source_snapshot.v1".into(),
        source_snapshot.iter().fold(String::with_capacity(64), |mut text, byte| {
            use std::fmt::Write;
            write!(text, "{byte:02x}").expect("writing to a String cannot fail");
            text
        }),
    );
    let artifact = antecedent_io::attach_prior_source_meta(artifact, &meta)
        .map_err(crate::CausalError::Serialization)?;
    let mut encoded = Vec::new();
    artifact.write_to(&mut encoded).map_err(crate::CausalError::Serialization)?;
    Ok(VerifiedBayesianPrior {
        envelope: std::sync::Arc::from(bytes),
        source_snapshot,
        posterior: encoded.into(),
    })
}
