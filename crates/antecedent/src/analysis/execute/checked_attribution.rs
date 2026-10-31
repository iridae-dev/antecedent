//! Builder-independent execution for frequentist DAG attribution operations.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, CausalPosterior, CausalQuery, Dag, DataInput, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, EffectEstimate, EstimatorId, ExecutionContext, GcmSlot,
    IdentificationResult, IdentifiedExecuteExtras, IdentifiedExecuteFinish,
    IdentifiedResultContext, IdentifierId, InferenceMode, Instant, OverlapPolicy,
    PhysicalExecutionPlan, Strategy, Study, StudyResult, TableView, TabularData, TimeSeriesData,
    VariableId, anomaly_attribution_query_with, bayesian_draw_count, elapsed_ns,
    finish_identified_execute_with_context, fit_gcm, nan_effect, parametric_scm_identification,
};
use crate::analysis::prepared::Inspect;
use crate::analysis::proof_support::proof_supplies_method_and_functional;
use crate::analysis::structure_route::{
    AttributionFit, AttributionOutcome, attribution_quantities_weighted, change_quantities,
    difference_measure,
};
use crate::gcm::FitResampling;

/// Refits an intrinsic-influence attribution runs for its posterior over the mechanisms when
/// the study does not set the draw count: each refit is a full nested attribution.
const INTRINSIC_POSTERIOR_DRAWS: usize = 100;

/// The refit populations of one replicate: row weights when Bayesian, else counts.
fn population_refit<'a>(
    bayesian: bool,
    baseline: &'a [f64],
    comparison: &'a [f64],
) -> antecedent_attribution::PopulationRefit<'a> {
    if bayesian {
        antecedent_attribution::PopulationRefit::RowWeights { baseline, comparison }
    } else {
        antecedent_attribution::PopulationRefit::Counts { baseline, comparison }
    }
}

/// Whether `query` is an intrinsic-influence attribution.
fn attribute_intrinsic_target(query: &CausalQuery) -> bool {
    matches!(
        query,
        CausalQuery::Attribute(attribute)
            if matches!(attribute.target, antecedent_core::AttributeTarget::IntrinsicInfluence(_))
    )
}

/// Whether `query` is an attribution or simulation the mechanism-refit resamplings cover: every
/// target of the fitted-model attribution and every simulation target, except mechanism-change
/// detection. A detection is a
/// two-sample test whose p-value is a tail probability of a null law of its statistic, not a
/// smooth functional of the refitted mechanisms, so it has neither a posterior over the
/// mechanisms nor a refit bootstrap; its Bayesian counterpart is the posterior probability of a
/// change under a declared prior.
fn attribute_refit_covered(query: &CausalQuery) -> bool {
    matches!(
        query,
        CausalQuery::Attribute(attribute)
            if !matches!(attribute.target, antecedent_core::AttributeTarget::MechanismChange { .. })
    ) || matches!(query, CausalQuery::Simulate(_))
}

/// Whether `query` is a mechanism-change detection, standalone or as an attribution target.
pub(crate) fn is_mechanism_change_query(query: &CausalQuery) -> bool {
    matches!(query, CausalQuery::MechanismChange(_))
        || matches!(
            query,
            CausalQuery::Attribute(attribute)
                if matches!(attribute.target, antecedent_core::AttributeTarget::MechanismChange { .. })
        )
}

/// Whether `query` is a mechanism-change detection of an attribution that declares the prior
/// over the change indicator, so its Bayesian reading (the posterior probability of a change)
/// has what it needs.
pub(crate) fn mechanism_change_posterior_declared(query: &CausalQuery) -> bool {
    matches!(
        query,
        CausalQuery::Attribute(attribute)
            if matches!(
                &attribute.target,
                antecedent_core::AttributeTarget::MechanismChange { query, .. }
                    if query.change_prior.is_some()
            )
    )
}

/// Whether `query` is a temporal rollout of the fitted dynamics, the one simulation target that
/// runs on the lag-aligned unfolding of a `TemporalDag`.
fn rollout_query(query: &CausalQuery) -> bool {
    matches!(
        query,
        CausalQuery::Simulate(simulate)
            if matches!(simulate.target, antecedent_core::SimulationTarget::TemporalRollout { .. })
    )
}

/// Whether a fitted-model target runs on the lag-aligned unfolding of a `TemporalDag`: every
/// attribution target the refit resamplings cover, with block weights or block counts keeping
/// the serial dependence of consecutive times inside each block. A simulation, a mechanism
/// change and a unit change as queries of their own stay on a static DAG.
fn attribute_on_unfolding(query: &CausalQuery) -> bool {
    (attribute_refit_covered(query) && !matches!(query, CausalQuery::Simulate(_)))
        || rollout_query(query)
        || matches!(query, CausalQuery::UnitChange(_))
        || is_mechanism_change_query(query)
}

/// Whether the fitted-model target `query` has a posterior over the mechanisms: every
/// attribution target except mechanism-change detection (see [`attribute_refit_covered`]).
/// A simulation reports the fitted model's answer.
pub(crate) fn attribute_posterior_supported(query: &CausalQuery) -> bool {
    attribute_refit_covered(query)
}

/// Whether a replicate count gives `query` the mechanism-refit row bootstrap: every covered
/// target but intrinsic influence, which carries its own `fit_bootstrap`.
pub(crate) fn attribute_bootstrap_supported(query: &CausalQuery) -> bool {
    attribute_refit_covered(query) && !attribute_intrinsic_target(query)
}

/// Whether a draw of `query` is dear: it reruns a nested simulation, a Shapley game over
/// features or units, or a hybrid refit per coalition, rather than reading the refit
/// mechanisms once.
fn attribute_draw_is_dear(query: &CausalQuery) -> bool {
    use antecedent_core::{AttributeTarget, ChangeEstimator};
    let CausalQuery::Attribute(attribute) = query else {
        return matches!(query, CausalQuery::Simulate(_));
    };
    !matches!(
        &attribute.target,
        AttributeTarget::Paths { .. }
            | AttributeTarget::Anomaly(_)
            | AttributeTarget::DistributionChange {
                estimator: ChangeEstimator::Sampled { .. },
                ..
            }
            | AttributeTarget::RootCauses { estimator: ChangeEstimator::Sampled { .. }, .. }
    )
}

/// Posterior draws of a fitted-model attribution: the study's draw count, except that a target
/// whose every draw is a full nested attribution or Shapley game runs a bounded number of
/// refits unless the caller set the count.
pub(crate) fn attribute_posterior_draws(
    query: &CausalQuery,
    config: &crate::BayesianConfig,
) -> usize {
    if attribute_draw_is_dear(query) && !config.n_draws_explicit {
        config.n_draws.min(INTRINSIC_POSTERIOR_DRAWS)
    } else {
        config.n_draws
    }
}

/// Whether an attribution entry runs the sealed operation for the uncertainty it carries: the
/// posterior over the mechanisms under Bayesian inference, and the mechanism-refit row
/// bootstrap when replicates are requested.
pub(crate) fn attribute_runs_checked(
    query: &CausalQuery,
    inference: &InferenceMode,
    bootstrap_replicates: u32,
) -> bool {
    match inference {
        InferenceMode::Bayesian(_) => {
            attribute_posterior_supported(query) || mechanism_change_posterior_declared(query)
        }
        InferenceMode::Frequentist => {
            bootstrap_replicates > 0 && attribute_bootstrap_supported(query)
        }
    }
}

#[derive(Clone, Debug)]
enum AttributionTarget {
    Anomaly(antecedent_core::AnomalyAttributionQuery),
    Change(antecedent_core::ChangeAttributionQuery),
    /// A fitted-model route (attribution, simulation, mechanism change or unit
    /// change) that runs on the mechanisms its program fits.
    Model(CausalQuery),
}

enum SampledWeights {
    Legacy(Vec<f64>),
    Run(antecedent_attribution::temporal_resampling::RunWeights),
}
impl std::ops::Deref for SampledWeights {
    type Target = [f64];
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Legacy(values) => values,
            Self::Run(values) => values,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum AttributionProcedure {
    Point,
    SharedDirichletRowWeights {
        draws: usize,
    },
    /// Frequentist whole-procedure multinomial row bootstrap (static DAG only).
    RowBootstrap {
        replicates: u32,
    },
}

/// A frozen GCM attribution target and its prepare-time identification contract.
#[derive(Clone)]
pub(crate) struct CheckedAttributionOperation {
    graph: Dag,
    target: AttributionTarget,
    cache: Arc<crate::analysis::prepared::CachedStaticIdentification>,
    inference: InferenceMode,
    procedure: AttributionProcedure,
    estimator: EstimatorId,
    physical: std::sync::Arc<PhysicalExecutionPlan>,
    result_context: IdentifiedResultContext,
    /// Present when the source graph was a `TemporalDag`. The static `graph` is its unfolding.
    lag_aligned: Option<super::temporal_attribution::LagAlignedScm>,
    temporal_input: Option<super::temporal_attribution::CompleteWindowInput>,
    temporal_budget: Option<antecedent_core::TemporalBudget>,
    /// Mechanism fit of a fitted-model route.
    mechanisms: Arc<crate::analysis::structure_route::MechanismProgram>,
    /// Present when the source graph was a mixed graph whose attribution is identified: the
    /// noise blocks (districts of the ancestral sub-graph) that are the Shapley players. The
    /// static `graph` is then the directed part of the mixed graph.
    blocks: Option<Arc<[Vec<VariableId>]>>,
}

impl std::fmt::Debug for CheckedAttributionOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckedAttributionOperation")
            .field("target", &self.target)
            .field("estimand", &self.cache.estimand)
            .field("identification_status", &self.cache.identification.status)
            .field("identifier", &IdentifierId::GcmParametric)
            .field("inference", &self.inference)
            .field("procedure", &self.procedure)
            .field("estimator", &self.estimator)
            .finish_non_exhaustive()
    }
}

impl CheckedAttributionOperation {
    pub(crate) fn checked(
        study: &Study,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        cache: &Arc<crate::analysis::prepared::CachedStaticIdentification>,
        ctx: &ExecutionContext,
    ) -> Result<Self, CausalError> {
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        Self::checked_within(study, physical, cache, &mut budget, ctx)
    }

    pub(super) fn checked_within(
        study: &Study,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        cache: &Arc<crate::analysis::prepared::CachedStaticIdentification>,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &ExecutionContext,
    ) -> Result<Self, CausalError> {
        Self::checked_on(study, physical, cache, attribution_graph(study, budget, ctx)?, None)
    }

    /// The operation on the directed part of an identified mixed graph, with its noise blocks
    /// as the Shapley players. `lag_aligned` is the unfolding when the mixed graph was temporal.
    pub(super) fn checked_blocks(
        study: &Study,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        cache: &Arc<crate::analysis::prepared::CachedStaticIdentification>,
        resolved: (Dag, Option<super::temporal_attribution::LagAlignedScm>),
        blocks: Vec<Vec<VariableId>>,
    ) -> Result<Self, CausalError> {
        Self::checked_on(study, physical, cache, resolved, Some(Arc::from(blocks)))
    }

    fn checked_on(
        study: &Study,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        cache: &Arc<crate::analysis::prepared::CachedStaticIdentification>,
        (graph, lag_aligned): (Dag, Option<super::temporal_attribution::LagAlignedScm>),
        blocks: Option<Arc<[Vec<VariableId>]>>,
    ) -> Result<Self, CausalError> {
        let target = match &study.query {
            CausalQuery::AnomalyAttribution(query) => {
                query.validate()?;
                AttributionTarget::Anomaly(query.clone())
            }
            CausalQuery::ChangeAttribution(query) => {
                query.validate()?;
                AttributionTarget::Change(query.clone())
            }
            query @ (CausalQuery::MechanismChange(_)
            | CausalQuery::UnitChange(_)
            | CausalQuery::Attribute(_)
            | CausalQuery::Simulate(_)) => {
                query.validate()?;
                AttributionTarget::Model(query.clone())
            }
            _ => {
                return Err(CausalError::Compile {
                    message: "checked attribution supports only anomaly, change and fitted-model \
                              targets"
                        .into(),
                });
            }
        };
        let fitted_model = matches!(target, AttributionTarget::Model(_));
        if blocks.is_some() && fitted_model {
            return Err(CausalError::Compile {
                message: "noise-block attribution supports only anomaly and change targets".into(),
            });
        }
        let target_query = target.query();
        if !study.declared.custom_validators.is_empty() {
            return Err(CausalError::Unsupported {
                message: "checked attribution does not support custom validators",
            });
        }
        if lag_aligned.is_some()
            && matches!(&study.query, CausalQuery::Simulate(_))
            && !rollout_query(&study.query)
        {
            return Err(crate::unsupported_reason!(
                "option_not_applicable",
                "on a TemporalDag a simulation is a temporal rollout of the fitted dynamics: \
                 draws, policy values and counterfactuals of the unfolded graph would sample \
                 the lagged copies as exogenous columns. Next attempt: use a TemporalRollout \
                 target, or fit the one-slice model on a supplied Dag"
            ));
        }
        let (estimator, procedure) = attribution_procedure(&target, study)?;
        // An accepted graph is the structure's provenance, not a different graph: the
        // mechanisms are fitted on it exactly as on a supplied one.
        let structure_ok = matches!(
            study.structure_source,
            crate::support::StructureSource::Explicit | crate::support::StructureSource::Accepted
        );
        if !structure_ok
            || (fitted_model && lag_aligned.is_some() && !attribute_on_unfolding(&target_query))
            || cache.identification.query != target_query
            || physical.logical.query != target_query
            || !physical.binding().names(IdentifierId::GcmParametric, estimator)
        {
            return Err(CausalError::Compile {
                message: "attribution target, source graph, identification, or estimator does not match the checked operation".into(),
            });
        }
        let estimand = &cache.estimand;
        if !proof_supplies_method_and_functional(&cache.identification, estimand) {
            return Err(CausalError::Compile {
                message: "attribution estimand is absent from its identification result".into(),
            });
        }
        Ok(Self {
            graph: graph.clone(),
            target,
            cache: Arc::clone(cache),
            inference: study.settings.inference.clone(),
            procedure,
            estimator,
            physical: physical.clone(),
            result_context: IdentifiedResultContext::from_study(study),
            lag_aligned,
            temporal_input: None,
            temporal_budget: None,
            mechanisms: study.mechanisms.clone().unwrap_or_else(|| {
                Arc::new(crate::analysis::structure_route::MechanismProgram::standard())
            }),
            blocks,
        })
    }

    /// The anomaly scores of `query` on `model`: one Shapley player per ancestor noise term, or
    /// per noise block on a mixed graph.
    fn score_anomalies(
        &self,
        model: &crate::gcm::CompiledCausalModel,
        data: &TabularData,
        query: &antecedent_core::AnomalyAttributionQuery,
        ctx: &ExecutionContext,
    ) -> Result<Vec<antecedent_attribution::AnomalyScores>, CausalError> {
        match &self.blocks {
            None => anomaly_attribution_query_with(model, data, query, ctx),
            Some(blocks) => {
                antecedent_attribution::score_anomalies_blocks_with(model, data, query, blocks, ctx)
                    .map_err(CausalError::from)
            }
        }
    }

    /// The distribution-change attribution of `query` on `model`, with both populations refit
    /// under `refit` when it is given: one Shapley player per mechanism, or per noise block on
    /// a mixed graph.
    fn change_attribution(
        &self,
        model: &crate::gcm::CompiledCausalModel,
        data: &TabularData,
        query: &antecedent_core::ChangeAttributionQuery,
        options: &antecedent_attribution::DistributionChangeOptions,
        refit: Option<antecedent_attribution::PopulationRefit<'_>>,
        ctx: &ExecutionContext,
    ) -> Result<
        antecedent_attribution::ChangeAttributionResult,
        antecedent_attribution::AttributionError,
    > {
        use antecedent_attribution::PopulationRefit;
        if let Some(blocks) = &self.blocks {
            return antecedent_attribution::distribution_change_blocks(
                model,
                data,
                query,
                options,
                blocks,
                refit.as_ref(),
                ctx,
            );
        }
        match refit {
            None => antecedent_attribution::distribution_change(model, data, query, options, ctx),
            Some(PopulationRefit::RowWeights { baseline, comparison }) => {
                antecedent_attribution::distribution_change_with_row_weights(
                    model, data, query, options, baseline, comparison, ctx,
                )
            }
            Some(PopulationRefit::Counts { baseline, comparison }) => {
                antecedent_attribution::distribution_change_with_frequency_weights(
                    model, data, query, options, baseline, comparison, ctx,
                )
            }
        }
    }

    /// Fit the frozen unfolding on a fresh lag-aligned table.
    ///
    /// # Errors
    ///
    /// The operation was sealed on a static DAG, or fitting and scoring fail.
    pub(crate) fn execute_series(
        &self,
        series: &TimeSeriesData,
        provenance: Option<&super::TemporalRowProvenance>,
        common_history: Option<u32>,
        shared_budget: Option<&mut antecedent_core::TemporalBudget>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let aligned = self.lag_aligned.as_ref().ok_or_else(|| CausalError::Compile {
            message: "temporal attribution lost its lag-aligned unfolding".into(),
        })?;
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "temporal attribution" });
        }
        let user_weighted = provenance.map_or_else(
            || series.storage().weights().is_some(),
            super::TemporalRowProvenance::user_sampling_weights,
        );
        if user_weighted {
            return Err(crate::unsupported_reason!(
                "option_not_applicable",
                "temporal attribution mechanism selection and fitting do not support user observation weights"
            ));
        }
        let mut local_budget = provenance.map(|p| p.budget().clone()).unwrap_or_else(|| {
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits())
        });
        let mut budget = shared_budget.unwrap_or(&mut local_budget);
        let input = super::temporal_attribution::CompleteWindowInput::build(
            aligned,
            series,
            common_history.unwrap_or_else(|| aligned.history()),
            provenance,
            &mut budget,
            ctx,
        )?;
        if self.result_context.query != self.query()
            || self.cache.identification.query != self.query()
            || !proof_supplies_method_and_functional(
                &self.cache.identification,
                &self.cache.estimand,
            )
        {
            return Err(CausalError::Compile {
                message: "attribution operation lost its checked target or identification binding"
                    .into(),
            });
        }
        let mut temporal_op = self.clone();
        temporal_op.temporal_input = Some(input.clone());
        temporal_op.temporal_budget = Some(budget.clone());
        let ticks = input.original_ticks();
        let source_bound = provenance.is_none().then_some(series.row_count());
        let mut result = match &self.target {
            AttributionTarget::Anomaly(query) => {
                let mut query = query.clone();
                let mut selector_lease = None;
                if let Some(requested) = &query.unit_rows {
                    if provenance.is_none() {
                        let count =
                            u64::try_from(requested.len()).map_err(|_| CausalError::Resource {
                                message: "anomaly row request size overflow".into(),
                            })?;
                        budget
                            .charge_work(count)
                            .map_err(|e| CausalError::Resource { message: e.to_string() })?;
                        let first_complete =
                            common_history.unwrap_or_else(|| aligned.history()) as usize;
                        for &row in requested.iter() {
                            if ctx.cancellation.is_cancelled() {
                                return Err(CausalError::Cancelled {
                                    stage: "temporal attribution",
                                });
                            }
                            if row < first_complete {
                                return Err(CausalError::Compile {
                                    message: format!(
                                        "anomaly requested original row {row} without complete observed history"
                                    ),
                                });
                            }
                        }
                    }
                    let selector = antecedent_core::PopulationSelector::Rows(requested.clone());
                    let (translated, lease) = Self::series_retained_selector(
                        &selector,
                        ticks,
                        source_bound,
                        &mut budget,
                        ctx,
                    )?;
                    let antecedent_core::PopulationSelector::Rows(rows) = translated else {
                        unreachable!()
                    };
                    query.unit_rows = Some(rows);
                    selector_lease = lease;
                }
                let _selector_lease = selector_lease;
                temporal_op.execute_anomaly(&query, input.table(), ctx, Some(ticks))
            }
            AttributionTarget::Change(query) => {
                let mut query = query.clone();
                let (baseline, _baseline_lease) = Self::series_retained_selector(
                    &query.baseline,
                    ticks,
                    source_bound,
                    &mut budget,
                    ctx,
                )?;
                let (comparison, _comparison_lease) = Self::series_retained_selector(
                    &query.comparison,
                    ticks,
                    source_bound,
                    &mut budget,
                    ctx,
                )?;
                query.baseline = baseline;
                query.comparison = comparison;
                temporal_op.execute_change(&query, input.table(), ctx)
            }
            AttributionTarget::Model(query) => {
                let needs_row_mapping = match query {
                    CausalQuery::MechanismChange(_) | CausalQuery::UnitChange(_) => true,
                    CausalQuery::Attribute(a) => matches!(
                        &a.target,
                        antecedent_core::AttributeTarget::DistributionChange { .. }
                            | antecedent_core::AttributeTarget::RootCauses { .. }
                            | antecedent_core::AttributeTarget::StructureChange { .. }
                            | antecedent_core::AttributeTarget::MechanismChange { .. }
                            | antecedent_core::AttributeTarget::Anomaly(_)
                            | antecedent_core::AttributeTarget::UnitChange(_)
                    ),
                    _ => false,
                };
                if needs_row_mapping {
                    let (query, _first_lease, _second_lease) = Self::series_model_query(
                        query,
                        ticks,
                        common_history.unwrap_or_else(|| aligned.history()),
                        source_bound,
                        &mut budget,
                        ctx,
                    )?;
                    temporal_op.execute_model(&query, input.table(), Some(&input), ctx)
                } else {
                    temporal_op.execute_model(query, input.table(), Some(&input), ctx)
                }
            }
        }?;
        result.diagnostics.push(Diagnostic::new(
            "gcm.attribution.complete_observed_windows_v2",
            DiagnosticKind::Execution,
            DiagnosticSeverity::Info,
            format!("fit and score use {} eligible complete observed windows from {} source rows with common history {}; source row coordinates are preserved. V2 resampling splits disconnected runs and derives independent segment streams; circular bootstrap segments wrap only within their run.", input.ticks().len(), series.row_count(), common_history.unwrap_or_else(|| aligned.history())),
        ));
        Ok(result)
    }

    pub(crate) fn query(&self) -> CausalQuery {
        self.target.query()
    }

    pub(crate) fn graph(&self) -> &Dag {
        &self.graph
    }

    pub(crate) fn identification(&self) -> &IdentificationResult {
        &self.cache.identification
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if self.result_context.query != self.query()
            || self.cache.identification.query != self.query()
            || !proof_supplies_method_and_functional(
                &self.cache.identification,
                &self.cache.estimand,
            )
        {
            return Err(CausalError::Compile {
                message: "attribution operation lost its checked target or identification binding"
                    .into(),
            });
        }
        // Rows hidden by an analysis mask (a refit on a subset of the rows) are deleted before
        // fitting and scoring. A mask the route ignored would make every subset refit
        // reproduce the full-sample vector, and the subset refuter could never see a change.
        if matches!(self.target, AttributionTarget::Anomaly(_) | AttributionTarget::Change(_)) {
            if let Some((table, kept)) = Self::visible_rows(data)? {
                return match &self.target {
                    AttributionTarget::Anomaly(query) => {
                        let query = antecedent_core::AnomalyAttributionQuery {
                            unit_rows: query
                                .unit_rows
                                .as_ref()
                                .map(|rows| Self::retained_positions(rows, &kept)),
                            ..query.clone()
                        };
                        self.execute_anomaly(&query, &table, ctx, Some(&kept))
                    }
                    AttributionTarget::Change(query) => {
                        let mut query = query.clone();
                        query.baseline = Self::retained_selector(&query.baseline, &kept);
                        query.comparison = Self::retained_selector(&query.comparison, &kept);
                        self.execute_change(&query, &table, ctx)
                    }
                    AttributionTarget::Model(_) => unreachable!("guarded above"),
                };
            }
        }
        match &self.target {
            AttributionTarget::Anomaly(query) => self.execute_anomaly(query, data, ctx, None),
            AttributionTarget::Change(query) => self.execute_change(query, data, ctx),
            AttributionTarget::Model(query) => self.execute_model(query, data, None, ctx),
        }
    }

    /// The table without its masked-out rows and the submitted positions of the rows it keeps,
    /// or `None` when no row is hidden.
    fn visible_rows(data: &TabularData) -> Result<Option<(TabularData, Vec<usize>)>, CausalError> {
        let Some(mask) = data.storage().analysis_mask().filter(|mask| !mask.is_all_valid()) else {
            return Ok(None);
        };
        let kept: Vec<usize> = (0..data.row_count()).filter(|&row| mask.is_valid(row)).collect();
        if kept.len() < 2 {
            return Err(CausalError::Compile {
                message: "an analysis mask leaves fewer than two rows to fit the mechanisms".into(),
            });
        }
        let storage = data.storage();
        let columns = (0..storage.schema().len())
            .map(|index| {
                storage
                    .column(VariableId::from_raw(u32::try_from(index).unwrap_or(u32::MAX)))?
                    .gather_rows(&kept)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let weights = storage
            .weights()
            .map(|weights| kept.iter().map(|&row| weights[row]).collect::<Arc<[f64]>>());
        let table = TabularData::new(antecedent_data::OwnedColumnarStorage::try_new(
            storage.schema().clone(),
            columns,
            None,
            weights,
        )?);
        Ok(Some((table, kept)))
    }

    /// Submitted row positions as positions in the retained rows; a hidden row drops out.
    fn retained_positions(rows: &[usize], kept: &[usize]) -> Arc<[usize]> {
        rows.iter().filter_map(|row| kept.binary_search(row).ok()).collect()
    }

    /// A population selector over the submitted rows as one over the retained rows.
    fn retained_selector(
        selector: &antecedent_core::PopulationSelector,
        kept: &[usize],
    ) -> antecedent_core::PopulationSelector {
        use antecedent_core::PopulationSelector;
        match selector {
            PopulationSelector::Rows(rows) => {
                PopulationSelector::Rows(Self::retained_positions(rows, kept))
            }
            PopulationSelector::TimeRange { start, end } => PopulationSelector::TimeRange {
                start: kept.partition_point(|&row| row < *start),
                end: kept.partition_point(|&row| row < *end),
            },
            other => other.clone(),
        }
    }

    fn series_model_query(
        query: &CausalQuery,
        ticks: &[usize],
        history: u32,
        source_bound: Option<usize>,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &ExecutionContext,
    ) -> Result<
        (
            CausalQuery,
            Option<antecedent_core::execution::AdmissionLease>,
            Option<antecedent_core::execution::AdmissionLease>,
        ),
        CausalError,
    > {
        use antecedent_core::{AttributeTarget, PopulationSelector};
        let mut query = query.clone();
        let mut first = None;
        let mut second = None;
        let mut change = |baseline: &mut PopulationSelector,
                          comparison: &mut PopulationSelector|
         -> Result<(), CausalError> {
            let (mapped, lease) =
                Self::series_retained_selector(baseline, ticks, source_bound, budget, ctx)?;
            *baseline = mapped;
            first = lease;
            let (mapped, lease) =
                Self::series_retained_selector(comparison, ticks, source_bound, budget, ctx)?;
            *comparison = mapped;
            second = lease;
            Ok(())
        };
        match &mut query {
            CausalQuery::MechanismChange(q) => change(&mut q.baseline, &mut q.comparison)?,
            CausalQuery::Attribute(q) => match &mut q.target {
                AttributeTarget::DistributionChange { change: c, .. }
                | AttributeTarget::RootCauses { change: c, .. }
                | AttributeTarget::StructureChange { change: c, .. } => {
                    change(&mut c.baseline, &mut c.comparison)?
                }
                AttributeTarget::MechanismChange { query: c, .. } => {
                    change(&mut c.baseline, &mut c.comparison)?
                }
                _ => {}
            },
            _ => {}
        }
        let unit_rows = match &mut query {
            CausalQuery::UnitChange(q) => Some(&mut q.unit_rows),
            CausalQuery::Attribute(q) => match &mut q.target {
                AttributeTarget::Anomaly(a) => Some(&mut a.unit_rows),
                AttributeTarget::UnitChange(a) => Some(&mut a.unit_rows),
                _ => None,
            },
            _ => None,
        };
        if let Some(Some(rows)) = unit_rows {
            let count = u64::try_from(rows.len()).map_err(|_| CausalError::Resource {
                message: "unit row request size overflow".into(),
            })?;
            budget
                .charge_work(count)
                .map_err(|e| CausalError::Resource { message: e.to_string() })?;
            if source_bound.is_some() {
                for &row in rows.iter() {
                    if ctx.cancellation.is_cancelled() {
                        return Err(CausalError::Cancelled { stage: "temporal attribution" });
                    }
                    if row < history as usize {
                        return Err(CausalError::Compile {
                            message: format!(
                                "requested original row {row} without complete observed history"
                            ),
                        });
                    }
                }
            }
            let (mapped, lease) = Self::series_retained_selector(
                &PopulationSelector::Rows(rows.clone()),
                ticks,
                source_bound,
                budget,
                ctx,
            )?;
            let PopulationSelector::Rows(mapped) = mapped else { unreachable!() };
            *rows = mapped;
            first = lease;
        }
        Ok((query, first, second))
    }

    fn retain_mapped_rows(
        &self,
        count: usize,
        ctx: &ExecutionContext,
    ) -> Result<Option<Arc<antecedent_core::execution::AdmissionLease>>, CausalError> {
        if count == 0 {
            return Ok(None);
        }
        let Some(shared) = self.temporal_budget.as_ref() else {
            return Err(CausalError::Compile {
                message: "temporal row mapping lost its budget".into(),
            });
        };
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "temporal attribution" });
        }
        let mut budget = shared.clone();
        let n = u64::try_from(count).map_err(|_| CausalError::Resource {
            message: "temporal result row count overflow".into(),
        })?;
        budget.charge_work(n).map_err(|e| CausalError::Resource { message: e.to_string() })?;
        let bytes = n.checked_mul(16).and_then(|v| v.checked_add(1024)).ok_or_else(|| {
            CausalError::Resource { message: "temporal result row bytes overflow".into() }
        })?;
        if bytes > budget.limits().max_bytes {
            return Err(CausalError::Resource { message: "temporal result row byte limit".into() });
        }
        let lease = ctx.acquire_admission(0, bytes).map_err(|e| match e {
            antecedent_core::execution::AdmissionError::Cancelled => {
                CausalError::Cancelled { stage: "temporal attribution" }
            }
            other => CausalError::Resource { message: other.to_string() },
        })?;
        Ok(Some(Arc::new(lease)))
    }

    fn model_row_count(slot: &GcmSlot) -> Result<usize, CausalError> {
        match slot {
            GcmSlot::Unit(result) => Ok(result.unit_rows.len()),
            GcmSlot::Attribution(payload) => match &payload.outcome {
                AttributionOutcome::Anomaly(scores) => {
                    scores.iter().try_fold(0_usize, |n, score| {
                        n.checked_add(score.rows.len()).ok_or_else(|| CausalError::Resource {
                            message: "temporal result row count overflow".into(),
                        })
                    })
                }
                AttributionOutcome::UnitChange(result) => Ok(result.unit_rows.len()),
                _ => Ok(0),
            },
            _ => Ok(0),
        }
    }

    fn remap_model_rows(
        slot: &mut GcmSlot,
        ticks: &[usize],
        ctx: &ExecutionContext,
    ) -> Result<(), CausalError> {
        fn remap(
            rows: &mut Arc<[usize]>,
            ticks: &[usize],
            ctx: &ExecutionContext,
        ) -> Result<(), CausalError> {
            for row in Arc::make_mut(rows).iter_mut() {
                if ctx.cancellation.is_cancelled() {
                    return Err(CausalError::Cancelled { stage: "temporal attribution" });
                }
                *row = *ticks.get(*row).ok_or_else(|| CausalError::Compile {
                    message: "attribution returned an invalid compact row".into(),
                })?;
            }
            Ok(())
        }
        match slot {
            GcmSlot::Unit(result) => remap(&mut result.unit_rows, ticks, ctx)?,
            GcmSlot::Attribution(payload) => match &mut payload.outcome {
                AttributionOutcome::Anomaly(scores) => {
                    for score in scores {
                        remap(&mut score.rows, ticks, ctx)?;
                    }
                }
                AttributionOutcome::UnitChange(result) => remap(&mut result.unit_rows, ticks, ctx)?,
                _ => {}
            },
            _ => {}
        }
        Ok(())
    }

    /// Translate original coordinates to compact positions, including every
    /// occurrence in a gathered refit. The lease remains live with the query.
    fn series_retained_selector(
        selector: &antecedent_core::PopulationSelector,
        kept: &[usize],
        source_bound: Option<usize>,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &ExecutionContext,
    ) -> Result<
        (antecedent_core::PopulationSelector, Option<antecedent_core::execution::AdmissionLease>),
        CausalError,
    > {
        use antecedent_core::PopulationSelector;
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "temporal attribution" });
        }
        if let (Some(n), PopulationSelector::TimeRange { end, .. }) = (source_bound, selector) {
            if *end > n {
                return Err(CausalError::Compile {
                    message: format!(
                        "temporal population range ends at {end}, beyond {n} source rows"
                    ),
                });
            }
        }
        let scans = match selector {
            PopulationSelector::Rows(rows) => rows.len().checked_mul(kept.len()),
            PopulationSelector::TimeRange { .. } => Some(kept.len()),
            _ => return Ok((selector.clone(), None)),
        }
        .ok_or_else(|| CausalError::Resource {
            message: "temporal selector work overflow".into(),
        })?;
        let work = u64::try_from(scans).map_err(|_| CausalError::Resource {
            message: "temporal selector work overflow".into(),
        })?;
        budget
            .charge_work(work)
            .map_err(|error| CausalError::Resource { message: error.to_string() })?;
        let mut count = 0_usize;
        match selector {
            PopulationSelector::Rows(rows) => {
                for row in rows.iter() {
                    if source_bound.is_some_and(|n| *row >= n) {
                        return Err(CausalError::Compile {
                            message: format!("temporal population row {row} is out of range"),
                        });
                    }
                    for tick in kept {
                        if ctx.cancellation.is_cancelled() {
                            return Err(CausalError::Cancelled { stage: "temporal attribution" });
                        }
                        if row == tick {
                            count += 1;
                        }
                    }
                }
            }
            PopulationSelector::TimeRange { start, end } => {
                for tick in kept {
                    if ctx.cancellation.is_cancelled() {
                        return Err(CausalError::Cancelled { stage: "temporal attribution" });
                    }
                    if *tick >= *start && *tick < *end {
                        count += 1;
                    }
                }
            }
            _ => unreachable!(),
        }
        let bytes = u64::try_from(count)
            .ok()
            .and_then(|n| n.checked_mul(16))
            .and_then(|n| n.checked_add(1024))
            .ok_or_else(|| CausalError::Resource {
                message: "temporal selector size overflow".into(),
            })?;
        if bytes > budget.limits().max_bytes {
            return Err(CausalError::Resource { message: "temporal selector byte limit".into() });
        }
        budget
            .charge_work(work)
            .map_err(|error| CausalError::Resource { message: error.to_string() })?;
        let lease = ctx.acquire_admission(0, bytes).map_err(|error| match error {
            antecedent_core::execution::AdmissionError::Cancelled => {
                CausalError::Cancelled { stage: "temporal attribution" }
            }
            other => CausalError::Resource { message: other.to_string() },
        })?;
        let mut positions = Vec::new();
        positions.try_reserve_exact(count).map_err(|_| CausalError::Resource {
            message: "temporal selector allocation refused".into(),
        })?;
        match selector {
            PopulationSelector::Rows(rows) => {
                for row in rows.iter() {
                    for (i, tick) in kept.iter().enumerate() {
                        if ctx.cancellation.is_cancelled() {
                            return Err(CausalError::Cancelled { stage: "temporal attribution" });
                        }
                        if row == tick {
                            positions.push(i);
                        }
                    }
                }
            }
            PopulationSelector::TimeRange { start, end } => {
                for (i, tick) in kept.iter().enumerate() {
                    if ctx.cancellation.is_cancelled() {
                        return Err(CausalError::Cancelled { stage: "temporal attribution" });
                    }
                    if *tick >= *start && *tick < *end {
                        positions.push(i);
                    }
                }
            }
            _ => unreachable!(),
        }
        Ok((PopulationSelector::Rows(positions.into()), Some(lease)))
    }

    /// Fit the program's mechanisms and run the fitted-model target on them, with the
    /// uncertainty its procedure carries: the posterior over the mechanisms under Bayesian
    /// inference, the mechanism-refit row (or block) bootstrap when replicates are requested,
    /// and the fitted model's point answer otherwise.
    fn execute_model(
        &self,
        query: &CausalQuery,
        data: &TabularData,
        observed: Option<&super::temporal_attribution::CompleteWindowInput>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let fit = self.model_fit(query, data.row_count());
        let change_query = match query {
            CausalQuery::MechanismChange(q) => Some(q),
            CausalQuery::Attribute(q) => match &q.target {
                antecedent_core::AttributeTarget::MechanismChange { query, .. } => Some(query),
                _ => None,
            },
            _ => None,
        };
        let (baseline_runs, comparison_runs) = if let Some(change) = change_query {
            let baseline = antecedent_attribution::resolve_rows(data, &change.baseline)
                .map_err(CausalError::rejected)?;
            let comparison = antecedent_attribution::resolve_rows(data, &change.comparison)
                .map_err(CausalError::rejected)?;
            (
                self.selected_temporal_runs(&baseline, ctx)?,
                self.selected_temporal_runs(&comparison, ctx)?,
            )
        } else {
            (None, None)
        };
        let reading = crate::analysis::structure_route::ChangeReading {
            block: self.lag_aligned.as_ref().map(|_| self.temporal_block_length(data.row_count())),
            bayesian: self.inference.is_bayesian() && is_mechanism_change_query(query),
            runs: baseline_runs
                .as_ref()
                .zip(comparison_runs.as_ref())
                .map(|(a, b)| (a.ranges(), b.ranges())),
            budget: self.temporal_budget.as_ref(),
        };
        let mut answer = super::structural_route::run_model_query_with_fit(
            &self.mechanisms,
            &self.graph,
            data,
            query,
            fit,
            super::structural_route::ModelRun {
                reading,
                unfolding: self.lag_aligned.as_ref(),
                observed,
                budget: self.temporal_budget.as_ref(),
            },
            ctx,
        )?;
        let mut posterior = None;
        match self.procedure {
            AttributionProcedure::SharedDirichletRowWeights { draws } => {
                let built = self.model_posterior(query, data, &answer, draws, observed, ctx)?;
                if let Some(index) = built.effect_column() {
                    answer.estimate.ate = built.summaries.mean[index];
                    answer.estimate.uncertainty.analytic.se = built.summaries.sd[index];
                }
                posterior = Some(built);
                answer.diagnostics.push(bayesian_attribution_diagnostic(&format!(
                    "Each draw refits every mechanism under {} (one weight vector per population \
                     for a change target), with the mechanism families held at the ones selected \
                     on the original rows, and recomputes the attribution on the refit: the \
                     posterior is over the mechanisms, given the empirical row law. The data \
                     references an attribution reads (feature means, parent means, anomaly \
                     reference) stay at their observed values. No coverage record measures this \
                     interval.",
                    self.weights_description(data.row_count()),
                )));
            }
            AttributionProcedure::RowBootstrap { replicates } => {
                self.model_bootstrap(query, data, replicates, &mut answer, observed, ctx)?;
            }
            AttributionProcedure::Point => {
                if self.inference.is_bayesian()
                    && !attribute_posterior_supported(query)
                    && !is_mechanism_change_query(query)
                {
                    answer.diagnostics.push(Diagnostic::new(
                        "gcm.attribution.posterior_not_built",
                        DiagnosticKind::Scientific,
                        DiagnosticSeverity::Warning,
                        "this target reports the fitted model's answer under Bayesian \
                         inference: it is a test or a simulation, not a smooth functional of \
                         the refitted mechanisms. Next attempt: declare a prior over the change \
                         indicator for the posterior probability of a mechanism change, or use \
                         the counterfactual for a simulation.",
                    ));
                }
            }
        }
        let row_lease = if let Some(input) = observed {
            let lease = self.retain_mapped_rows(Self::model_row_count(&answer.slot)?, ctx)?;
            Self::remap_model_rows(&mut answer.slot, input.original_ticks(), ctx)?;
            lease
        } else {
            None
        };
        let (treatment, outcome) = super::gcm_query_vars(query)?;
        let mut result = finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish::new(
                &self.physical,
                self.cache.identification.clone(),
                self.cache.estimand.clone(),
                answer.estimate,
                Strategy::new(IdentifierId::GcmParametric, self.estimator),
                treatment,
                outcome,
            )
            .identify_cached(true)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                diagnostics: Some(answer.diagnostics),
                posterior,
                n_draws: self.published_draws(),
                gcm: Some(answer.slot),
                empty_provenance: true,
                ..Default::default()
            }),
        );
        if let Some(lease) = row_lease {
            result.retained_resources.push(lease);
        }
        Ok(result)
    }

    /// The weights one draw puts on the rows, in words.
    fn weights_description(&self, n: usize) -> String {
        match &self.lag_aligned {
            Some(_) => format!(
                "run-aware v2 Exp(1) weights for blocks of {} retained rows, split into independently weighted contiguous source-run segments",
                self.temporal_block_length(n)
            ),
            None => "one Dirichlet(1,…,1) row-weight vector".to_owned(),
        }
    }

    /// The mechanism-fit refits an intrinsic-influence target runs under this procedure.
    ///
    /// On a lag-aligned unfolding the refits resample or weight blocks of consecutive times, so
    /// the serial dependence of the series is kept inside each block.
    fn model_fit(&self, query: &CausalQuery, n: usize) -> Option<AttributionFit> {
        if !attribute_intrinsic_target(query) {
            return None;
        }
        let block = self.lag_aligned.as_ref().map(|_| self.temporal_block_length(n));
        match (self.procedure.clone(), block) {
            (AttributionProcedure::SharedDirichletRowWeights { draws }, block) => {
                Some(AttributionFit {
                    replicates: u32::try_from(draws).unwrap_or(u32::MAX),
                    resampling: block.map_or(FitResampling::DirichletRowWeights, |block| {
                        FitResampling::DirichletBlockWeights { block }
                    }),
                })
            }
            (_, Some(block)) => match query {
                CausalQuery::Attribute(attribute) => match &attribute.target {
                    antecedent_core::AttributeTarget::IntrinsicInfluence(target)
                        if target.fit_bootstrap_replicates > 0 =>
                    {
                        Some(AttributionFit {
                            replicates: target.fit_bootstrap_replicates,
                            resampling: FitResampling::BlockBootstrap { block },
                        })
                    }
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        }
    }

    /// The posterior over the mechanisms of a fitted-model target: the Dirichlet-weighted
    /// refits of the attribution, or the refit contributions the intrinsic-influence engine
    /// returned.
    fn model_posterior(
        &self,
        query: &CausalQuery,
        data: &TabularData,
        answer: &super::structural_route::ModelAnswer,
        draws: usize,
        observed: Option<&super::temporal_attribution::CompleteWindowInput>,
        ctx: &ExecutionContext,
    ) -> Result<CausalPosterior, CausalError> {
        let (columns, has_primary_effect) = if attribute_intrinsic_target(query) {
            (intrinsic_posterior_columns(&answer.slot)?, false)
        } else {
            let replicates = self.attribute_replicates(query, data, draws, true, observed, ctx)?;
            let primary = replicates.names.first().is_some_and(|name| name == "total_change");
            (replicates.columns(), primary)
        };
        super::attribution_path::attribution_posterior(
            columns,
            self.cache.identification.required_assumptions.clone(),
            self.cache.identification.status,
            super::ATTRIBUTE_DIRICHLET_BACKEND,
            has_primary_effect,
        )
    }

    /// The mechanism-refit bootstrap of a fitted-model target: the standard error of the
    /// primary quantity on the estimate, and every quantity's standard error in a diagnostic.
    fn model_bootstrap(
        &self,
        query: &CausalQuery,
        data: &TabularData,
        replicates: u32,
        answer: &mut super::structural_route::ModelAnswer,
        observed: Option<&super::temporal_attribution::CompleteWindowInput>,
        ctx: &ExecutionContext,
    ) -> Result<(), CausalError> {
        let draws = usize::try_from(replicates).unwrap_or(usize::MAX);
        let outcome = self.attribute_replicates(query, data, draws, false, observed, ctx)?;
        let columns = outcome.columns();
        let accounted: Vec<_> = columns
            .iter()
            .map(|(_, values)| super::gcm_bootstrap::accounting(values, outcome.failed, replicates))
            .collect();
        let Some(primary) = accounted.first().copied() else {
            return Ok(());
        };
        answer.estimate.uncertainty.bootstrap = primary;
        let target = columns.first().map_or("the attribution", |(name, _)| name.as_str());
        let mut diagnostic = self.bootstrap_disclosure(
            &format!("{target} of the fitted-model attribution"),
            data.row_count(),
            &mut answer.estimate,
        );
        let fields: Vec<(String, String)> = columns
            .iter()
            .zip(&accounted)
            .map(|((name, _), accounting)| {
                (
                    format!("bootstrap_se.{name}"),
                    accounting
                        .se
                        .map_or_else(|| "unquantified".to_owned(), |se| format!("{se:.6e}")),
                )
            })
            .chain([
                ("replicates_ok".to_owned(), primary.replicates_ok.unwrap_or(0).to_string()),
                (
                    "replicates_failed".to_owned(),
                    primary.replicates_failed.unwrap_or(0).to_string(),
                ),
                ("calibration".to_owned(), "unmeasured".to_owned()),
            ])
            .collect();
        diagnostic = diagnostic.with_fields(fields);
        answer.diagnostics.push(diagnostic);
        Ok(())
    }

    /// Recompute a fitted-model attribution on `replicates` refits of the mechanisms: each
    /// under fresh Dirichlet (row or block) weights when `bayesian`, else fresh bootstrap
    /// counts. The mechanism families are the ones selected on the original rows, as on the
    /// anomaly and change cells.
    ///
    /// A failed replicate is an error under weights and failed mass under counts; it is never
    /// redrawn.
    fn attribute_replicates(
        &self,
        query: &CausalQuery,
        data: &TabularData,
        replicates: usize,
        bayesian: bool,
        observed: Option<&super::temporal_attribution::CompleteWindowInput>,
        ctx: &ExecutionContext,
    ) -> Result<Replicates, CausalError> {
        let (attribute, simulate) = match query {
            CausalQuery::Attribute(attribute) => (Some(attribute), None),
            CausalQuery::Simulate(simulate) => (None, Some(simulate)),
            _ => {
                return Err(CausalError::Compile {
                    message: "a mechanism-refit replicate needs an attribution or rollout query"
                        .into(),
                });
            }
        };
        let fitted = self.mechanisms.fit(&self.graph, data)?;
        let mut stream =
            ctx.rng.stream_for(antecedent_core::StreamDomain::Attribution, 0xA77B_7A7E);
        let mut out = Replicates { names: Vec::new(), rows: Vec::new(), failed: 0 };
        let mut admitted = None;
        for replicate in 0..replicates {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled {
                    stage: super::super::stage::STAGE_UNCERTAINTY,
                });
            }
            let mut counts_stream;
            let rng = if bayesian {
                &mut stream
            } else {
                counts_stream = super::gcm_bootstrap::replicate_stream(
                    ctx,
                    0xA77B_B007,
                    u32::try_from(replicate).unwrap_or(u32::MAX),
                );
                &mut counts_stream
            };
            let replicate = match (attribute, simulate) {
                (Some(attribute), _) => self.attribute_replicate(
                    attribute,
                    &fitted,
                    data,
                    bayesian,
                    rng,
                    replicate as u64,
                    ctx,
                ),
                (None, Some(simulate)) => self.simulate_replicate(
                    simulate,
                    &fitted,
                    data,
                    bayesian,
                    rng,
                    replicate as u64,
                    observed,
                    ctx,
                ),
                (None, None) => unreachable!("a replicate query is an attribution or a rollout"),
            };
            match replicate {
                Ok(values) => {
                    if out.names.is_empty() {
                        out.names = values.iter().map(|(name, _)| name.clone()).collect();
                        admitted = Some(admit_posterior_columns(ctx, replicates, out.names.len())?);
                    } else if values.len() != out.names.len()
                        || values.iter().zip(&out.names).any(|((name, _), first)| name != first)
                    {
                        return Err(CausalError::Compile {
                            message: "a mechanism-refit replicate changed the attribution's \
                                      components"
                                .into(),
                        });
                    }
                    out.rows.push(values.into_iter().map(|(_, value)| value).collect());
                }
                Err(_) if ctx.cancellation.is_cancelled() => {
                    return Err(CausalError::Cancelled {
                        stage: super::super::stage::STAGE_UNCERTAINTY,
                    });
                }
                Err(error @ (CausalError::Cancelled { .. } | CausalError::Resource { .. }))
                    if self.temporal_input.is_some() =>
                {
                    return Err(error);
                }
                Err(error) if bayesian => return Err(error),
                Err(_) => out.failed += 1,
            }
        }
        drop(admitted);
        if out.rows.len() < 2 && bayesian {
            return Err(CausalError::Compile {
                message: "Bayesian attribution draws have inconsistent shape".into(),
            });
        }
        Ok(out)
    }

    /// One replicate of a temporal rollout: the mechanisms refit under fresh block weights (or
    /// block counts), the dynamics rolled forward from the last rows of the series, and the mean
    /// of every variable at every step read off the rollout.
    fn simulate_replicate(
        &self,
        simulate: &antecedent_core::SimulateQuery,
        fitted: &crate::gcm::FittedGcm,
        data: &TabularData,
        bayesian: bool,
        rng: &mut antecedent_core::CausalRng,
        replicate: u64,
        observed: Option<&super::temporal_attribution::CompleteWindowInput>,
        ctx: &ExecutionContext,
    ) -> Result<Vec<(String, f64)>, CausalError> {
        let n = data.row_count();
        let weights = if bayesian {
            self.draw_row_weights(n, rng, self.full_temporal_runs(), replicate, ctx)?
        } else {
            self.draw_row_counts(n, rng, self.full_temporal_runs(), replicate, ctx)?
        };
        let registry = crate::gcm::MechanismRegistry::standard();
        let cancelled = || ctx.cancellation.is_cancelled();
        let store = if bayesian {
            registry.refit_weighted_polled(
                &fitted.model,
                data,
                &fitted.assignments,
                &weights,
                &cancelled,
            )
        } else {
            registry.refit_frequency_weighted_polled(
                &fitted.model,
                data,
                &fitted.assignments,
                &weights,
                &cancelled,
            )
        }
        .map_err(|error| {
            if self.temporal_input.is_some() {
                CausalError::from(error)
            } else {
                CausalError::rejected(error)
            }
        })?;
        let refit = crate::gcm::FittedGcm {
            model: fitted.model.clone().with_mechanisms(store),
            assignments: fitted.assignments.clone(),
        };
        let outcome = match &self.lag_aligned {
            Some(aligned) => super::structural_route::rollout_on_unfolding(
                &refit,
                data,
                aligned,
                observed,
                self.temporal_budget.as_ref(),
                simulate,
                ctx,
            )?,
            None => crate::analysis::structure_route::run_simulate(&refit, data, simulate, ctx)?,
        };
        let nodes = self.lag_aligned.as_ref().map_or(
            refit.model.n_nodes(),
            super::temporal_attribution::LagAlignedScm::contemporaneous_count,
        );
        let mut quantities = crate::analysis::structure_route::simulation_quantities(
            &outcome, nodes,
        )
        .ok_or_else(|| CausalError::Compile {
            message: "this simulation has no mechanism-refit replicate".into(),
        })?;
        // The means above reuse the query's seed in every refit (common random numbers keep the
        // refit spread free of Monte Carlo noise). The predictive law of a path needs fresh
        // innovations in every refit, so one path is rolled out under a seed drawn per replicate.
        if let antecedent_core::SimulationTarget::TemporalRollout {
            schedule,
            n_rows,
            n_steps,
            window,
            history,
            ..
        } = &simulate.target
        {
            let single = antecedent_core::SimulateQuery::new(
                antecedent_core::SimulationTarget::TemporalRollout {
                    schedule: schedule.clone(),
                    // A supplied history has one value per row, so it keeps its row count.
                    n_rows: if history.is_empty() { 1 } else { *n_rows },
                    n_steps: *n_steps,
                    window: *window,
                    history: history.clone(),
                    seed: rng.next_u64(),
                },
            );
            let path_outcome = match &self.lag_aligned {
                Some(aligned) => super::structural_route::rollout_on_unfolding(
                    &refit,
                    data,
                    aligned,
                    observed,
                    self.temporal_budget.as_ref(),
                    &single,
                    ctx,
                )?,
                None => crate::analysis::structure_route::run_simulate(&refit, data, &single, ctx)?,
            };
            quantities.extend(
                crate::analysis::structure_route::simulated_path_quantities(&path_outcome, nodes)
                    .ok_or_else(|| CausalError::Compile {
                    message: "a rollout replicate has no simulated path".into(),
                })?,
            );
        }
        Ok(quantities)
    }

    /// One replicate of a fitted-model attribution on mechanisms refit under fresh weights.
    fn attribute_replicate(
        &self,
        attribute: &antecedent_core::AttributeQuery,
        fitted: &crate::gcm::FittedGcm,
        data: &TabularData,
        bayesian: bool,
        rng: &mut antecedent_core::CausalRng,
        replicate: u64,
        ctx: &ExecutionContext,
    ) -> Result<Vec<(String, f64)>, CausalError> {
        use antecedent_core::{AttributeTarget, ChangeEstimator};
        let draw = |op: &Self,
                    n: usize,
                    positions: Option<&[usize]>,
                    rng: &mut antecedent_core::CausalRng,
                    key: u64| {
            let partition =
                positions.map(|rows| op.selected_temporal_runs(rows, ctx)).transpose()?.flatten();
            let runs = partition
                .as_ref()
                .map(|p| p.ranges())
                .or_else(|| if positions.is_none() { op.full_temporal_runs() } else { None });
            if bayesian {
                op.draw_row_weights(n, rng, runs, key, ctx)
            } else {
                op.draw_row_counts(n, rng, runs, key, ctx)
            }
        };
        match &attribute.target {
            AttributeTarget::DistributionChange { change, estimator }
            | AttributeTarget::RootCauses { change, estimator } => {
                let baseline_rows = antecedent_attribution::resolve_rows(data, &change.baseline)
                    .map_err(CausalError::rejected)?;
                let comparison_rows =
                    antecedent_attribution::resolve_rows(data, &change.comparison)
                        .map_err(CausalError::rejected)?;
                let baseline = draw(
                    self,
                    baseline_rows.len(),
                    Some(&baseline_rows),
                    rng,
                    antecedent_core::derive_stream(replicate, 0),
                )?;
                let comparison = draw(
                    self,
                    comparison_rows.len(),
                    Some(&comparison_rows),
                    rng,
                    antecedent_core::derive_stream(replicate, 1),
                )?;
                let refit = population_refit(bayesian, &baseline, &comparison);
                let result = match estimator {
                    ChangeEstimator::Sampled { measure, n_samples, seed } => {
                        let options = antecedent_attribution::DistributionChangeOptions {
                            measure: difference_measure(*measure),
                            n_samples: *n_samples,
                            seed: *seed,
                        };
                        if bayesian {
                            antecedent_attribution::distribution_change_with_row_weights(
                                &fitted.model,
                                data,
                                change,
                                &options,
                                &baseline,
                                &comparison,
                                ctx,
                            )
                        } else {
                            antecedent_attribution::distribution_change_with_frequency_weights(
                                &fitted.model,
                                data,
                                change,
                                &options,
                                &baseline,
                                &comparison,
                                ctx,
                            )
                        }
                    }
                    ChangeEstimator::Robust { max_rows, n_samples, seed } => {
                        antecedent_attribution::distribution_change_robust_with_refit(
                            &fitted.model,
                            data,
                            change,
                            &antecedent_attribution::RobustChangeOptions {
                                max_rows: *max_rows,
                                n_samples: *n_samples,
                                seed: *seed,
                            },
                            &refit,
                            ctx,
                        )
                    }
                }
                .map_err(CausalError::rejected)?;
                Ok(change_quantities(&result))
            }
            AttributeTarget::StructureChange { change, comparison, measure, n_samples, seed } => {
                let baseline_rows = antecedent_attribution::resolve_rows(data, &change.baseline)
                    .map_err(CausalError::rejected)?;
                let comparison_rows =
                    antecedent_attribution::resolve_rows(data, &change.comparison)
                        .map_err(CausalError::rejected)?;
                let baseline = draw(
                    self,
                    baseline_rows.len(),
                    Some(&baseline_rows),
                    rng,
                    antecedent_core::derive_stream(replicate, 0),
                )?;
                let comparison_weights = draw(
                    self,
                    comparison_rows.len(),
                    Some(&comparison_rows),
                    rng,
                    antecedent_core::derive_stream(replicate, 1),
                )?;
                let refit = population_refit(bayesian, &baseline, &comparison_weights);
                let comparison_model =
                    crate::analysis::structure_route::structure_comparison_model(
                        &self.graph,
                        comparison,
                    )?;
                let result = antecedent_attribution::structure_change_with_refit(
                    &fitted.model,
                    &comparison_model,
                    data,
                    change,
                    &antecedent_attribution::StructureChangeOptions {
                        measure: difference_measure(*measure),
                        n_samples: *n_samples,
                        seed: *seed,
                    },
                    &refit,
                    ctx,
                )
                .map_err(CausalError::rejected)?;
                Ok(change_quantities(&result))
            }
            AttributeTarget::MechanismChange { .. } => Err(CausalError::Unsupported {
                message: "mechanism-change detection is a test, not a smooth functional of the \
                          refitted mechanisms",
            }),
            AttributeTarget::IntrinsicInfluence(_) => Err(CausalError::Compile {
                message: "intrinsic influence draws its refits inside the engine".into(),
            }),
            _ => {
                // Paths, feature relevance, anomaly scores and unit change read the fitted
                // mechanisms (and, for the last three, the rows): refit them on the whole
                // table under one weight vector and recompute.
                let weights = draw(self, data.row_count(), None, rng, replicate)?;
                self.whole_table_replicate(attribute, fitted, data, bayesian, &weights, ctx)
            }
        }
    }

    /// The quantities of an attribution recomputed on mechanisms refit on the whole table under
    /// one weight vector.
    fn whole_table_replicate(
        &self,
        attribute: &antecedent_core::AttributeQuery,
        fitted: &crate::gcm::FittedGcm,
        data: &TabularData,
        bayesian: bool,
        weights: &[f64],
        ctx: &ExecutionContext,
    ) -> Result<Vec<(String, f64)>, CausalError> {
        let registry = crate::gcm::MechanismRegistry::standard();
        let store = if bayesian {
            registry.refit_weighted_polled(
                &fitted.model,
                data,
                &fitted.assignments,
                weights,
                &|| ctx.cancellation.is_cancelled(),
            )
        } else {
            registry.refit_frequency_weighted_polled(
                &fitted.model,
                data,
                &fitted.assignments,
                weights,
                &|| ctx.cancellation.is_cancelled(),
            )
        }
        .map_err(|error| {
            if self.temporal_input.is_some() {
                CausalError::from(error)
            } else {
                CausalError::rejected(error)
            }
        })?;
        let refit = crate::gcm::FittedGcm {
            model: fitted.model.clone().with_mechanisms(store),
            assignments: fitted.assignments.clone(),
        };
        let outcome = crate::analysis::structure_route::run_attribute_with_fit(
            &refit,
            &self.graph,
            data,
            attribute,
            None,
            crate::analysis::structure_route::ChangeReading::default(),
            None,
            None,
            ctx,
        )?;
        attribution_quantities_weighted(&outcome, weights).ok_or_else(|| CausalError::Compile {
            message: "this attribution has no mechanism-refit replicate".into(),
        })
    }

    /// Draw count published on the result: absent for point procedures.
    fn published_draws(&self) -> Option<u32> {
        match self.procedure {
            AttributionProcedure::SharedDirichletRowWeights { draws } => {
                Some(u32::try_from(draws).unwrap_or(u32::MAX))
            }
            AttributionProcedure::Point | AttributionProcedure::RowBootstrap { .. } => None,
        }
    }

    /// One frequentist replicate's row counts over `n` rows: multinomial single rows on a
    /// static table, circular blocks of consecutive times on a lag-aligned unfolding.
    fn draw_row_counts(
        &self,
        n: usize,
        rng: &mut antecedent_core::CausalRng,
        runs: Option<&[std::ops::Range<usize>]>,
        replicate: u64,
        ctx: &ExecutionContext,
    ) -> Result<SampledWeights, CausalError> {
        match (&self.lag_aligned, runs, &self.temporal_budget) {
            (Some(_), Some(runs), Some(budget)) => {
                let mut budget = budget.clone();
                antecedent_attribution::temporal_resampling::bootstrap_run_counts(
                    n,
                    self.temporal_block_length(n),
                    runs,
                    replicate,
                    rng,
                    &mut budget,
                    ctx,
                )
                .map(SampledWeights::Run)
                .map_err(CausalError::from)
            }
            (Some(_), None, _) => Err(CausalError::Compile {
                message: "temporal resampling lost its row partition".into(),
            }),
            (None, _, _) => {
                Ok(SampledWeights::Legacy(super::gcm_bootstrap::bootstrap_counts(rng, n)))
            }
            _ => Err(CausalError::Compile {
                message: "temporal resampling lost its shared budget".into(),
            }),
        }
    }

    fn selected_temporal_runs(
        &self,
        positions: &[usize],
        ctx: &ExecutionContext,
    ) -> Result<Option<super::temporal_attribution::SelectedRuns>, CausalError> {
        match (&self.temporal_input, &self.temporal_budget) {
            (Some(input), Some(budget)) => {
                let mut budget = budget.clone();
                Ok(Some(input.selected_runs(positions, &mut budget, ctx)?))
            }
            (None, None) => Ok(None),
            _ => Err(CausalError::Compile {
                message: "temporal resampling lost its row owner".into(),
            }),
        }
    }

    fn temporal_block_length(&self, n: usize) -> usize {
        self.temporal_input.as_ref().map_or_else(
            || self.lag_aligned.as_ref().map_or(1, |aligned| aligned.block_length(n)),
            |input| input.block_length(n),
        )
    }

    fn full_temporal_runs(&self) -> Option<&[std::ops::Range<usize>]> {
        self.temporal_input.as_ref().map(|input| input.runs())
    }

    /// The bootstrap disclosure of `target` over `n` rows, recording the row-block
    /// family on `estimate` when the rows are one lag-aligned series.
    fn bootstrap_disclosure(
        &self,
        target: &str,
        n: usize,
        estimate: &mut EffectEstimate,
    ) -> Diagnostic {
        let accounting = estimate.uncertainty.bootstrap;
        match &self.lag_aligned {
            Some(_) => {
                estimate.uncertainty.block.family =
                    Some(antecedent_estimate::CircularBlockFamily::RowBlock);
                super::gcm_bootstrap::bootstrap_diagnostic_over(
                    target,
                    &format!(
                        "v2 run-aware circular-block bootstrap (nominal block {}; disconnected segments use derived streams and wrap only within their own run)",
                        self.temporal_block_length(n)
                    ),
                    &accounting,
                )
            }
            None => super::gcm_bootstrap::bootstrap_diagnostic(target, &accounting),
        }
    }

    /// One draw's row weights over `n` rows: Dirichlet row weights on a static table,
    /// block weights on the consecutive times of a lag-aligned unfolding.
    fn draw_row_weights(
        &self,
        n: usize,
        rng: &mut antecedent_core::CausalRng,
        runs: Option<&[std::ops::Range<usize>]>,
        replicate: u64,
        ctx: &ExecutionContext,
    ) -> Result<SampledWeights, CausalError> {
        match (&self.lag_aligned, runs, &self.temporal_budget) {
            (Some(_), Some(runs), Some(budget)) => {
                let mut budget = budget.clone();
                antecedent_attribution::temporal_resampling::bayesian_run_weights(
                    n,
                    self.temporal_block_length(n),
                    runs,
                    replicate,
                    rng,
                    &mut budget,
                    ctx,
                )
                .map(SampledWeights::Run)
                .map_err(CausalError::from)
            }
            (Some(_), None, _) => Err(CausalError::Compile {
                message: "temporal resampling lost its row partition".into(),
            }),
            (None, _, _) => {
                Ok(SampledWeights::Legacy(super::attribution_path::dirichlet_row_weights(n, rng)))
            }
            _ => Err(CausalError::Compile {
                message: "temporal resampling lost its shared budget".into(),
            }),
        }
    }

    /// Fits the GCM, scores every row against the query reference, and publishes the anomaly attribution.
    fn execute_anomaly(
        &self,
        query: &antecedent_core::AnomalyAttributionQuery,
        data: &TabularData,
        ctx: &ExecutionContext,
        submitted_rows: Option<&[usize]>,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let fitted = fit_gcm(self.graph.clone(), data)?;
        // A series request names original ticks, translated by the entry path.
        // Keep the existing static scoring convention unchanged.
        let scorer_query = if self.lag_aligned.is_some() {
            query.clone()
        } else {
            antecedent_core::AnomalyAttributionQuery::new(query.targets.clone(), query.max_units)
                .with_reference(query.reference)
        };
        let scores = self.score_anomalies(&fitted.model, data, &scorer_query, ctx)?;
        let outcome = *query.targets.first().ok_or_else(|| CausalError::Compile {
            message: "checked anomaly target has no outcome variable".into(),
        })?;
        let posterior = self.anomaly_posterior(query, data, &fitted, &scores, outcome, ctx)?;
        let (estimate, bootstrap_diagnostic) =
            self.anomaly_bootstrap(query, data, &fitted, &scores, ctx)?;
        let (estimate, posterior, mut extra_diagnostics) =
            publish_lag_aligned(self.lag_aligned.is_some(), estimate, posterior);
        extra_diagnostics.extend(bootstrap_diagnostic);
        // Scored rows are positions in the retained table; publish the submitted positions.
        let row_lease = if submitted_rows.is_some() && self.temporal_input.is_some() {
            let count = scores.iter().try_fold(0_usize, |n, score| {
                n.checked_add(score.rows.len()).ok_or_else(|| CausalError::Resource {
                    message: "temporal result row count overflow".into(),
                })
            })?;
            self.retain_mapped_rows(count, ctx)?
        } else {
            None
        };
        let scores = match submitted_rows {
            None => scores,
            Some(kept) => scores
                .into_iter()
                .map(|mut score| {
                    let rows = score
                        .rows
                        .iter()
                        .map(|&row| {
                            if ctx.cancellation.is_cancelled() {
                                return Err(CausalError::Cancelled {
                                    stage: "temporal attribution",
                                });
                            }
                            kept.get(row).copied().ok_or_else(|| CausalError::Compile {
                                message: "anomaly returned an invalid compact row".into(),
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    score.rows = rows.into();
                    Ok(score)
                })
                .collect::<Result<Vec<_>, CausalError>>()?,
        };
        if posterior.is_some() {
            let disclosure = if self.temporal_input.is_some() {
                match query.reference {
                    antecedent_core::AnomalyReference::Empirical => {
                        "Posterior uses v2 run-aware block weights on the observed complete windows, splitting disconnected source runs into independently derived segments. The empirical marginal reference is held fixed at the observed median and MAD; the interval is conditional on that realized reference."
                    }
                    antecedent_core::AnomalyReference::Fixed { .. } => {
                        "Posterior uses v2 run-aware block weights on the observed complete windows, splitting disconnected source runs into independently derived segments. The declared fixed marginal reference remains fixed across draws."
                    }
                }
            } else {
                match query.reference {
                    antecedent_core::AnomalyReference::Empirical => {
                        "Posterior is the Bayesian bootstrap of the mean anomaly score: each draw reweights the per-row scores by one shared Dirichlet(1,…,1) row-weight vector while holding the observed marginal reference (median / 1.4826·MAD) fixed, so the interval is for the mean relative to that realized sample's reference, not a fixed population functional."
                    }
                    antecedent_core::AnomalyReference::Fixed { .. } => {
                        "Posterior is the Bayesian bootstrap of the population mean anomaly score against the injected fixed reference: each draw reweights the per-row scores by one shared Dirichlet(1,…,1) row-weight vector, so the interval is a credible interval for the fixed functional μ_A = E_Y[−log 2Φ(−|Y − center|/scale)]."
                    }
                }
            };
            extra_diagnostics.push(bayesian_attribution_diagnostic(disclosure));
        }
        let mut result = finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish::new(
                &self.physical,
                self.cache.identification.clone(),
                self.cache.estimand.clone(),
                estimate,
                Strategy::new(IdentifierId::GcmParametric, self.estimator),
                outcome,
                outcome,
            )
            .identify_cached(true)
            .extra_diagnostics(extra_diagnostics)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                posterior,
                n_draws: self.published_draws(),
                gcm: Some(GcmSlot::Anomaly(scores)),
                empty_provenance: true,
                ..Default::default()
            }),
        );
        if let Some(lease) = row_lease {
            result.retained_resources.push(lease);
        }
        Ok(result)
    }

    /// Fits the GCM and publishes the distribution-change attribution with its optional posterior.
    fn execute_change(
        &self,
        query: &antecedent_core::ChangeAttributionQuery,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let fitted = fit_gcm(self.graph.clone(), data)?;
        let change = self
            .change_attribution(
                &fitted.model,
                data,
                query,
                &antecedent_attribution::DistributionChangeOptions::default(),
                None,
                ctx,
            )
            .map_err(CausalError::from)?;
        let mut estimate = EffectEstimate::new(
            change.total_change,
            f64::NAN,
            antecedent_core::AssumptionSet::default(),
            OverlapPolicy::ExplicitOverride,
        );
        let options = antecedent_attribution::DistributionChangeOptions::default();
        let posterior = self.change_posterior(query, data, &fitted.model, &options, ctx)?;
        let bootstrap_diagnostic = self
            .change_bootstrap(query, data, &fitted.model, &options, ctx)?
            .map(|(accounting, diagnostic)| {
                estimate.uncertainty.bootstrap = accounting;
                if self.lag_aligned.is_some() {
                    estimate.uncertainty.block.family =
                        Some(antecedent_estimate::CircularBlockFamily::RowBlock);
                }
                diagnostic
            });
        if let Some(posterior) = &posterior {
            if let Some(index) = posterior.effect_column() {
                estimate.ate = posterior.summaries.mean[index];
                estimate.uncertainty.analytic.se = posterior.summaries.sd[index];
            }
        }
        let (estimate, posterior, mut extra_diagnostics) =
            publish_lag_aligned(self.lag_aligned.is_some(), estimate, posterior);
        extra_diagnostics.extend(bootstrap_diagnostic);
        if posterior.is_some() {
            extra_diagnostics.push(bayesian_attribution_diagnostic("Each population has one Dirichlet(1,…,1) row-weight vector per draw shared across mechanisms; total and Shapley components use the same weighted attribution."));
        }
        Ok(finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish::new(
                &self.physical,
                self.cache.identification.clone(),
                self.cache.estimand.clone(),
                estimate,
                Strategy::new(IdentifierId::GcmParametric, self.estimator),
                query.outcome,
                query.outcome,
            )
            .identify_cached(true)
            .extra_diagnostics(extra_diagnostics)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                posterior,
                n_draws: self.published_draws(),
                gcm: Some(GcmSlot::Change(change)),
                empty_provenance: true,
                ..Default::default()
            }),
        ))
    }

    /// Frequentist mechanism-refit bootstrap of the first target's mean anomaly score:
    /// the point is that mean and the interval its bootstrap standard error.
    fn anomaly_bootstrap(
        &self,
        query: &antecedent_core::AnomalyAttributionQuery,
        data: &TabularData,
        fitted: &crate::gcm::FittedGcm,
        observed_scores: &[antecedent_attribution::AnomalyScores],
        ctx: &ExecutionContext,
    ) -> Result<(EffectEstimate, Option<Diagnostic>), CausalError> {
        let AttributionProcedure::RowBootstrap { replicates } = self.procedure else {
            return Ok((nan_effect(), None));
        };
        let Some(primary) = observed_scores.first() else {
            return Ok((nan_effect(), None));
        };
        let n = data.row_count();
        let n_rows = n.checked_mul(2).ok_or_else(|| CausalError::Resource {
            message: "anomaly bootstrap row count overflow".to_owned(),
        })?;
        let _admitted = super::gcm_bootstrap::admit_replicates(ctx, replicates, 1, n_rows)?;
        let unit = vec![1.0_f64; n];
        let point =
            super::attribution_path::dirichlet_weighted_mean(&primary.scores, &primary.rows, &unit);
        let registry = crate::gcm::MechanismRegistry::standard();
        let scorer_query = if self.temporal_input.is_some() {
            query.clone()
        } else {
            antecedent_core::AnomalyAttributionQuery::new(query.targets.clone(), query.max_units)
                .with_reference(query.reference)
        };
        let cancelled = || ctx.cancellation.is_cancelled();
        let mut values = Vec::with_capacity(usize::try_from(replicates).unwrap_or(0));
        let mut failed = 0_u32;
        for replicate in 0..replicates {
            if cancelled() {
                return Err(CausalError::Cancelled {
                    stage: super::super::stage::STAGE_UNCERTAINTY,
                });
            }
            let mut rng = super::gcm_bootstrap::replicate_stream(ctx, 0xA110_B007, replicate);
            let counts = self.draw_row_counts(
                n,
                &mut rng,
                self.full_temporal_runs(),
                u64::from(replicate),
                ctx,
            )?;
            let store = match registry.refit_frequency_weighted_polled(
                &fitted.model,
                data,
                &fitted.assignments,
                &counts,
                &cancelled,
            ) {
                Ok(store) => store,
                Err(_) if cancelled() => {
                    return Err(CausalError::Cancelled {
                        stage: super::super::stage::STAGE_UNCERTAINTY,
                    });
                }
                Err(error @ antecedent_model::ModelError::ResourceLimit { .. })
                    if self.temporal_input.is_some() =>
                {
                    return Err(CausalError::from(error));
                }
                Err(_) => {
                    failed += 1;
                    continue;
                }
            };
            match self.score_anomalies(
                &fitted.model.clone().with_mechanisms(store),
                data,
                &scorer_query,
                ctx,
            ) {
                Ok(draw_scores) => {
                    // The resampled population's mean score: the replicate's row counts
                    // weight the per-row scores, as the Dirichlet weights do on the
                    // Bayesian cell.
                    let value = draw_scores
                        .iter()
                        .find(|score| score.target == primary.target)
                        .map_or(f64::NAN, |score| {
                            super::attribution_path::dirichlet_weighted_mean(
                                &score.scores,
                                &score.rows,
                                &counts,
                            )
                        });
                    if self.temporal_input.is_some() && !value.is_finite() {
                        failed += 1;
                    } else {
                        values.push(value);
                    }
                }
                Err(_) if cancelled() => {
                    return Err(CausalError::Cancelled {
                        stage: super::super::stage::STAGE_UNCERTAINTY,
                    });
                }
                Err(error @ (CausalError::Cancelled { .. } | CausalError::Resource { .. }))
                    if self.temporal_input.is_some() =>
                {
                    return Err(error);
                }
                Err(_) => failed += 1,
            }
        }
        let accounting = super::gcm_bootstrap::accounting(&values, failed, replicates);
        let mut estimate = EffectEstimate::new(
            point,
            f64::NAN,
            antecedent_core::AssumptionSet::default(),
            OverlapPolicy::ExplicitOverride,
        );
        estimate.uncertainty.bootstrap = accounting;
        let diagnostic = self.bootstrap_disclosure(
            &format!("mean anomaly score of target v{}", primary.target.raw()),
            n,
            &mut estimate,
        );
        Ok((estimate, Some(diagnostic)))
    }

    /// Frequentist mechanism-refit bootstrap of the total distribution change: each
    /// replicate draws row counts within each population and refits both.
    fn change_bootstrap(
        &self,
        query: &antecedent_core::ChangeAttributionQuery,
        data: &TabularData,
        model: &crate::gcm::CompiledCausalModel,
        options: &antecedent_attribution::DistributionChangeOptions,
        ctx: &ExecutionContext,
    ) -> Result<Option<(antecedent_estimate::BootstrapAccounting, Diagnostic)>, CausalError> {
        let AttributionProcedure::RowBootstrap { replicates } = self.procedure else {
            return Ok(None);
        };
        let baseline_rows = antecedent_attribution::resolve_rows(data, &query.baseline)
            .map_err(CausalError::rejected)?;
        let comparison_rows = antecedent_attribution::resolve_rows(data, &query.comparison)
            .map_err(CausalError::rejected)?;
        let baseline_n = baseline_rows.len();
        let comparison_n = comparison_rows.len();
        let baseline_runs = self.selected_temporal_runs(&baseline_rows, ctx)?;
        let comparison_runs = self.selected_temporal_runs(&comparison_rows, ctx)?;
        let rows = baseline_n.checked_add(comparison_n).ok_or_else(|| CausalError::Resource {
            message: "change bootstrap row count overflow".to_owned(),
        })?;
        let _admitted = super::gcm_bootstrap::admit_replicates(ctx, replicates, 1, rows)?;
        let mut totals = Vec::with_capacity(usize::try_from(replicates).unwrap_or(0));
        let mut failed = 0_u32;
        for replicate in 0..replicates {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled {
                    stage: super::super::stage::STAGE_UNCERTAINTY,
                });
            }
            let mut rng = super::gcm_bootstrap::replicate_stream(ctx, 0xC4A6_B007, replicate);
            let baseline_counts = self.draw_row_counts(
                baseline_n,
                &mut rng,
                baseline_runs.as_ref().map(|runs| runs.ranges()),
                antecedent_core::derive_stream(u64::from(replicate), 0),
                ctx,
            )?;
            let comparison_counts = self.draw_row_counts(
                comparison_n,
                &mut rng,
                comparison_runs.as_ref().map(|runs| runs.ranges()),
                antecedent_core::derive_stream(u64::from(replicate), 1),
                ctx,
            )?;
            match self.change_attribution(
                model,
                data,
                query,
                options,
                Some(antecedent_attribution::PopulationRefit::Counts {
                    baseline: &baseline_counts,
                    comparison: &comparison_counts,
                }),
                ctx,
            ) {
                Ok(draw) => totals.push(draw.total_change),
                Err(_) if ctx.cancellation.is_cancelled() => {
                    return Err(CausalError::Cancelled {
                        stage: super::super::stage::STAGE_UNCERTAINTY,
                    });
                }
                Err(
                    error @ (antecedent_attribution::AttributionError::Cancelled
                    | antecedent_attribution::AttributionError::Budget { .. }),
                ) if self.temporal_input.is_some() => return Err(CausalError::from(error)),
                Err(antecedent_attribution::AttributionError::Model(
                    error @ antecedent_model::ModelError::ResourceLimit { .. },
                )) if self.temporal_input.is_some() => return Err(CausalError::from(error)),
                Err(_) => failed += 1,
            }
        }
        let accounting = super::gcm_bootstrap::accounting(&totals, failed, replicates);
        let diagnostic = match &self.lag_aligned {
            Some(_) => super::gcm_bootstrap::bootstrap_diagnostic_over(
                "total distribution change",
                &format!(
                    "v2 run-aware circular-block bootstrap (nominal block {}; disconnected segments use derived streams and wrap only within their own run)",
                    self.temporal_block_length(baseline_n.min(comparison_n))
                ),
                &accounting,
            ),
            None => {
                super::gcm_bootstrap::bootstrap_diagnostic("total distribution change", &accounting)
            }
        };
        Ok(Some((accounting, diagnostic)))
    }

    fn anomaly_posterior(
        &self,
        query: &antecedent_core::AnomalyAttributionQuery,
        data: &TabularData,
        fitted: &crate::gcm::FittedGcm,
        observed_scores: &[antecedent_attribution::AnomalyScores],
        outcome: VariableId,
        ctx: &ExecutionContext,
    ) -> Result<Option<CausalPosterior>, CausalError> {
        let AttributionProcedure::SharedDirichletRowWeights { draws } = self.procedure else {
            return Ok(None);
        };
        let _admitted = admit_posterior_columns(ctx, draws, observed_scores.len())?;
        let mut columns = observed_scores
            .iter()
            .map(|score| {
                (format!("mean_anomaly_score[{}]", score.target.raw()), Vec::with_capacity(draws))
            })
            .collect::<Vec<_>>();
        let registry = crate::gcm::MechanismRegistry::standard();
        let scorer_query = if self.temporal_input.is_some() {
            query.clone()
        } else {
            antecedent_core::AnomalyAttributionQuery::new(query.targets.clone(), query.max_units)
                .with_reference(query.reference)
        };
        let mut rng = ctx.rng.stream_for(antecedent_core::StreamDomain::Attribution, 0xA110_7A7E);
        for replicate in 0..draws {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled {
                    stage: super::super::stage::STAGE_ESTIMATE_POINT,
                });
            }
            let weights = self.draw_row_weights(
                data.row_count(),
                &mut rng,
                self.full_temporal_runs(),
                replicate as u64,
                ctx,
            )?;
            let store = registry
                .refit_weighted(&fitted.model, data, &fitted.assignments, &weights)
                .map_err(|error| {
                    if self.temporal_input.is_some() {
                        CausalError::from(error)
                    } else {
                        CausalError::rejected(error)
                    }
                })?;
            let draw_scores = self.score_anomalies(
                &fitted.model.clone().with_mechanisms(store),
                data,
                &scorer_query,
                ctx,
            )?;
            if draw_scores.len() != columns.len() {
                return Err(CausalError::Compile {
                    message: "Bayesian anomaly draw changed target set".into(),
                });
            }
            for (column, score) in columns.iter_mut().zip(draw_scores) {
                // Bayesian bootstrap of the mean anomaly score under the shared Dirichlet
                // row weights, for both references: the factual scores do not depend on the
                // mechanism refit, so an unweighted mean would be identical every draw (a
                // degenerate posterior). See `execute_anomaly` for the full rationale.
                column.1.push(super::attribution_path::dirichlet_weighted_mean(
                    &score.scores,
                    &score.rows,
                    &weights,
                ));
            }
        }
        let identification = if self.temporal_input.is_some() {
            self.cache.identification.clone()
        } else {
            parametric_scm_identification(
                CausalQuery::AnomalyAttribution(query.clone()),
                outcome,
                outcome,
            )
            .0
        };
        Ok(Some(super::attribution_path::attribution_posterior(
            columns,
            identification.required_assumptions,
            identification.status,
            super::ANOMALY_DIRICHLET_BACKEND,
            false,
        )?))
    }

    fn change_posterior(
        &self,
        query: &antecedent_core::ChangeAttributionQuery,
        data: &TabularData,
        model: &crate::gcm::CompiledCausalModel,
        options: &antecedent_attribution::DistributionChangeOptions,
        ctx: &ExecutionContext,
    ) -> Result<Option<CausalPosterior>, CausalError> {
        let AttributionProcedure::SharedDirichletRowWeights { draws } = self.procedure else {
            return Ok(None);
        };
        let posterior_columns =
            self.change_posterior_columns(query, data, model, options, draws, ctx)?;
        let identification = if self.temporal_input.is_some() {
            self.cache.identification.clone()
        } else {
            parametric_scm_identification(
                CausalQuery::ChangeAttribution(query.clone()),
                query.outcome,
                query.outcome,
            )
            .0
        };
        Ok(Some(super::attribution_path::attribution_posterior(
            posterior_columns,
            identification.required_assumptions,
            identification.status,
            super::CHANGE_DIRICHLET_BACKEND,
            true,
        )?))
    }

    /// The posterior columns of a Dirichlet row-weight bootstrap of the distribution change:
    /// the total first, then one column per component, one weight vector per population per
    /// draw.
    fn change_posterior_columns(
        &self,
        query: &antecedent_core::ChangeAttributionQuery,
        data: &TabularData,
        model: &crate::gcm::CompiledCausalModel,
        options: &antecedent_attribution::DistributionChangeOptions,
        draws: usize,
        ctx: &ExecutionContext,
    ) -> Result<Vec<(String, Vec<f64>)>, CausalError> {
        let baseline_rows = antecedent_attribution::resolve_rows(data, &query.baseline)
            .map_err(CausalError::rejected)?;
        let comparison_rows = antecedent_attribution::resolve_rows(data, &query.comparison)
            .map_err(CausalError::rejected)?;
        let baseline_n = baseline_rows.len();
        let comparison_n = comparison_rows.len();
        let baseline_runs = self.selected_temporal_runs(&baseline_rows, ctx)?;
        let comparison_runs = self.selected_temporal_runs(&comparison_rows, ctx)?;
        let mut rng = ctx.rng.stream_for(antecedent_core::StreamDomain::Attribution, 0xC4A6_7A7E);
        let _totals_admitted = admit_posterior_columns(ctx, draws, 1)?;
        let mut component_admitted = None;
        let mut components: Option<Vec<(String, Vec<f64>)>> = None;
        let mut totals = Vec::with_capacity(draws);
        for replicate in 0..draws {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled {
                    stage: super::super::stage::STAGE_ESTIMATE_POINT,
                });
            }
            let baseline_weights = self.draw_row_weights(
                baseline_n,
                &mut rng,
                baseline_runs.as_ref().map(|runs| runs.ranges()),
                antecedent_core::derive_stream(replicate as u64, 0),
                ctx,
            )?;
            let comparison_weights = self.draw_row_weights(
                comparison_n,
                &mut rng,
                comparison_runs.as_ref().map(|runs| runs.ranges()),
                antecedent_core::derive_stream(replicate as u64, 1),
                ctx,
            )?;
            let draw = self
                .change_attribution(
                    model,
                    data,
                    query,
                    options,
                    Some(antecedent_attribution::PopulationRefit::RowWeights {
                        baseline: &baseline_weights,
                        comparison: &comparison_weights,
                    }),
                    ctx,
                )
                .map_err(CausalError::rejected)?;
            totals.push(draw.total_change);
            let columns = components.get_or_insert_with(|| {
                draw.contributions
                    .iter()
                    .map(|c| {
                        (
                            format!("component[{}]", c.component.variable().raw()),
                            Vec::with_capacity(draws),
                        )
                    })
                    .collect()
            });
            if draw.contributions.len() != columns.len() {
                return Err(CausalError::Compile {
                    message: "Bayesian change attribution draw changed component set".into(),
                });
            }
            if component_admitted.is_none() {
                // The component count is known once the first draw returns.
                component_admitted = Some(admit_posterior_columns(ctx, draws, columns.len())?);
            }
            for (column, component) in columns.iter_mut().zip(draw.contributions.iter()) {
                column.1.push(component.contribution);
            }
        }
        let mut posterior_columns = vec![("total_change".to_owned(), totals)];
        posterior_columns.extend(components.unwrap_or_default());
        Ok(posterior_columns)
    }
}

/// The per-source posterior columns of an intrinsic-influence attribution run under
/// Dirichlet row-weight refits: the contribution of each source in every refit.
fn intrinsic_posterior_columns(slot: &GcmSlot) -> Result<Vec<(String, Vec<f64>)>, CausalError> {
    let GcmSlot::Attribution(payload) = slot else {
        return Err(CausalError::Compile {
            message: "an intrinsic-influence posterior needs the attribution it was run on".into(),
        });
    };
    let AttributionOutcome::IntrinsicInfluence(result) = &payload.outcome else {
        return Err(CausalError::Compile {
            message: "an intrinsic-influence posterior needs an intrinsic-influence outcome".into(),
        });
    };
    if result.fit_draws.len() < 2 {
        return Err(crate::unsupported_reason!(
            "option_not_applicable",
            "the posterior over the mechanisms of an intrinsic-influence attribution needs its \
             mechanism refits, and too few completed (the planned simulation exceeds the draw \
             ceiling, or a refit failed). Next attempt: lower the outer, inner or replicate \
             draws, or request fewer draws, so every refit fits the budget"
        ));
    }
    Ok(result
        .sources
        .iter()
        .enumerate()
        .map(|(index, source)| {
            (
                format!("source[{}]", source.source.raw()),
                result.fit_draws.iter().map(|draw| draw[index]).collect(),
            )
        })
        .collect())
}

/// The replicates of a mechanism-refit resampling: the quantities' names, one row of values per
/// completed replicate, and the replicates that failed.
struct Replicates {
    names: Vec<String>,
    rows: Vec<Vec<f64>>,
    failed: u32,
}

impl Replicates {
    /// The values of each quantity across the completed replicates.
    fn columns(&self) -> Vec<(String, Vec<f64>)> {
        self.names
            .iter()
            .enumerate()
            .map(|(index, name)| (name.clone(), self.rows.iter().map(|row| row[index]).collect()))
            .collect()
    }
}

/// Lease `draws * columns` posterior values on the pool before the columns are filled.
fn admit_posterior_columns(
    ctx: &ExecutionContext,
    draws: usize,
    columns: usize,
) -> Result<antecedent_core::execution::AdmissionLease, CausalError> {
    let bytes = draws
        .checked_mul(columns)
        .and_then(|cells| cells.checked_mul(size_of::<f64>()))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| CausalError::Resource {
            message: "attribution posterior size overflow".to_owned(),
        })?;
    ctx.acquire_admission(0, bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => {
            CausalError::Cancelled { stage: super::super::stage::STAGE_ESTIMATE_POINT }
        }
        other => CausalError::Resource { message: other.to_string() },
    })
}

/// A `TemporalDag` publishes the point, and for Bayesian the posterior mean, with no interval.
fn publish_lag_aligned(
    lag_aligned: bool,
    mut estimate: EffectEstimate,
    posterior: Option<CausalPosterior>,
) -> (EffectEstimate, Option<CausalPosterior>, Vec<Diagnostic>) {
    if !lag_aligned {
        return (estimate, posterior, Vec::new());
    }
    let mut diagnostics = vec![Diagnostic::new(
        "gcm.attribution.lag_aligned",
        DiagnosticKind::Execution,
        DiagnosticSeverity::Info,
        "attribution ran on the complete observed lag windows of the TemporalDag unfolding",
    )];
    if let Some(posterior) = &posterior {
        let mean = posterior.summaries.mean.first().copied().unwrap_or(f64::NAN);
        if mean.is_finite() {
            estimate.ate = mean;
            estimate.uncertainty.analytic.se =
                posterior.summaries.sd.first().copied().unwrap_or(f64::NAN);
        }
        diagnostics.push(Diagnostic::new(
            "gcm.attribution.block_bayesian_bootstrap",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "v2 block weights split each drawn block at disconnected source runs and derive independent segment streams; every mechanism is refit under those weights; no coverage record measures this interval",
        ));
    }
    (estimate, posterior, diagnostics)
}

fn bayesian_attribution_diagnostic(detail: &str) -> Diagnostic {
    Diagnostic::new(
        "gcm.attribution.bayesian",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "Posterior is conditional on mechanism-family selection from the original unweighted data. {detail} It does not address model misspecification or causal identification."
        ),
    )
}

impl AttributionTarget {
    fn query(&self) -> CausalQuery {
        match self {
            Self::Anomaly(query) => CausalQuery::AnomalyAttribution(query.clone()),
            Self::Change(query) => CausalQuery::ChangeAttribution(query.clone()),
            Self::Model(query) => query.clone(),
        }
    }
}

/// Read-only target and procedure retained by a checked GCM attribution plan.
#[derive(Clone, Debug)]
pub struct CheckedAttributionInfo {
    /// Exact anomaly or distribution-change target.
    pub query: CausalQuery,
    /// Prepare-time identifier and estimator.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Prepare-time estimator.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Canonical supplied DAG edges.
    pub graph_edges: std::sync::Arc<[(u32, u32)]>,
    /// Identification status fixed at preparation.
    pub identification_status: antecedent_identify::IdentificationStatus,
}

impl Inspect for CheckedAttributionOperation {
    type Info = CheckedAttributionInfo;

    fn info(&self) -> CheckedAttributionInfo {
        let edges = self.graph().directed_edge_pairs_sorted();
        CheckedAttributionInfo {
            query: self.query(),
            identifier: IdentifierId::GcmParametric,
            estimator: self.estimator,
            graph_edges: edges.into(),
            identification_status: self.identification().status,
        }
    }
}

/// Resolves the static DAG, unfolding a `TemporalDag` over the supplied series when the source graph is temporal.
fn attribution_graph(
    study: &Study,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<(Dag, Option<super::temporal_attribution::LagAlignedScm>), CausalError> {
    Ok(if let Some(dag) = study.graph.as_dag() {
        (dag.clone(), None)
    } else if let Some(temporal) = study.graph.as_temporal_dag() {
        let (DataInput::Temporal(series) | DataInput::Event(series)) = &study.data else {
            return Err(CausalError::Unsupported {
                message: "temporal attribution requires a time series",
            });
        };
        let aligned = super::temporal_attribution::LagAlignedScm::unfold_within(
            temporal, series, budget, ctx,
        )?;
        let dag = aligned.dag.clone();
        (dag, Some(aligned))
    } else {
        return Err(CausalError::Unsupported {
            message: "checked attribution requires a supplied DAG or TemporalDag",
        });
    })
}

/// Selects the estimator and inference procedure for the target, refusing prior-bearing Bayesian configurations.
fn attribution_procedure(
    target: &AttributionTarget,
    study: &Study,
) -> Result<(EstimatorId, AttributionProcedure), CausalError> {
    Ok(match (target, &study.settings.inference) {
        // The posterior over the mechanisms of a fitted-model target: Dirichlet row-weight
        // refits (block weights on a lag-aligned unfolding) of the mechanisms, the families
        // held at the ones selected on the data.
        (AttributionTarget::Model(query), InferenceMode::Bayesian(config))
            if attribute_posterior_supported(query) =>
        {
            if config.prior.is_some()
                || config.prior_artifact.is_some()
                || config.external_compose.is_some()
            {
                return Err(CausalError::Unsupported {
                    message: "Bayesian attribution uses its shared-row Bayesian bootstrap and does not accept coefficient priors or external prior composition",
                });
            }
            bayesian_draw_count(&study.settings.inference)?;
            let draws = attribute_posterior_draws(query, config);
            (EstimatorId::GcmFitBayesian, AttributionProcedure::SharedDirichletRowWeights { draws })
        }
        // The mechanism-refit bootstrap of a fitted-model target: multinomial rows of
        // independent subjects on a static DAG, circular blocks of consecutive times on a
        // lag-aligned TemporalDag unfolding.
        (AttributionTarget::Model(query), InferenceMode::Frequentist)
            if study.settings.bootstrap_replicates > 0 && attribute_bootstrap_supported(query) =>
        {
            if study.graph.as_dag().is_some() {
                super::gcm_bootstrap::require_subject_resampling(study.declared.sampling_design)?;
            } else {
                super::gcm_bootstrap::require_series_resampling(study.declared.sampling_design)?;
            }
            (
                EstimatorId::GcmFit,
                AttributionProcedure::RowBootstrap {
                    replicates: study.settings.bootstrap_replicates,
                },
            )
        }
        // Every other fitted-model target reports the fitted model's answer.
        (AttributionTarget::Model(_), _) => (EstimatorId::GcmFit, AttributionProcedure::Point),
        (_, InferenceMode::Frequentist) => {
            // A requested replicate count is the mechanism-refit bootstrap: multinomial
            // rows of independent subjects on a static DAG, circular blocks of consecutive
            // times on a lag-aligned TemporalDag unfolding.
            if study.settings.bootstrap_replicates > 0 && study.graph.as_dag().is_some() {
                super::gcm_bootstrap::require_subject_resampling(study.declared.sampling_design)?;
                (
                    EstimatorId::GcmFit,
                    AttributionProcedure::RowBootstrap {
                        replicates: study.settings.bootstrap_replicates,
                    },
                )
            } else if study.settings.bootstrap_replicates > 0
                && matches!(study.graph.class(), crate::accepted::GraphClass::TemporalDag)
            {
                super::gcm_bootstrap::require_series_resampling(study.declared.sampling_design)?;
                (
                    EstimatorId::GcmFit,
                    AttributionProcedure::RowBootstrap {
                        replicates: study.settings.bootstrap_replicates,
                    },
                )
            } else {
                (EstimatorId::GcmFit, AttributionProcedure::Point)
            }
        }
        (AttributionTarget::Anomaly(_), InferenceMode::Bayesian(config)) => {
            if config.prior.is_some()
                || config.prior_artifact.is_some()
                || config.external_compose.is_some()
            {
                return Err(CausalError::Unsupported {
                    message: "Bayesian attribution uses its shared-row Bayesian bootstrap and does not accept coefficient priors or external prior composition",
                });
            }
            (
                EstimatorId::GcmFitBayesian,
                AttributionProcedure::SharedDirichletRowWeights {
                    draws: bayesian_draw_count(&study.settings.inference)?,
                },
            )
        }
        (AttributionTarget::Change(_), InferenceMode::Bayesian(config)) => {
            if config.prior.is_some()
                || config.prior_artifact.is_some()
                || config.external_compose.is_some()
            {
                return Err(CausalError::Unsupported {
                    message: "Bayesian attribution uses its shared-row Bayesian bootstrap and does not accept coefficient priors or external prior composition",
                });
            }
            (
                EstimatorId::GcmAttributionBayesian,
                AttributionProcedure::SharedDirichletRowWeights {
                    draws: bayesian_draw_count(&study.settings.inference)?,
                },
            )
        }
    })
}

#[cfg(test)]
mod bayesian_checked_attribution_tests {
    use super::*;
    use crate::analysis::builder::RefuteSuite;
    use antecedent_core::{AnomalyAttributionQuery, CausalQuery, PopulationSelector, VariableId};
    use antecedent_data::TabularData;
    use antecedent_graph::DenseNodeId;

    fn context() -> ExecutionContext {
        ExecutionContext::for_tests(0xBA71_5EED)
    }

    fn anomaly_fixture() -> (TabularData, Dag) {
        let x: Vec<f64> = (0..20).map(f64::from).collect();
        let y: Vec<f64> =
            (0..20).map(|i| if i == 19 { 200.0 } else { 2.0 * f64::from(i) }).collect();
        let data = TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())])
            .expect("fixture data");
        let mut graph = Dag::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        (data, graph)
    }

    fn change_fixture() -> (TabularData, Dag) {
        let x: Vec<f64> = (0..80).map(|i| f64::from(i % 40) * 0.1).collect();
        let y: Vec<f64> = (0..80)
            .map(|i| {
                let x = f64::from(i % 40) * 0.1;
                if i < 40 { 1.0 + 2.0 * x } else { 6.0 + 2.0 * x }
            })
            .collect();
        let data = TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())])
            .expect("fixture data");
        let mut graph = Dag::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        (data, graph)
    }

    fn checked_operation(
        data: &TabularData,
        graph: Dag,
        query: &CausalQuery,
        estimator: EstimatorId,
    ) -> CheckedAttributionOperation {
        let ctx = context();
        let builder = crate::Study::tabular(data.clone())
            .graph(graph)
            .query(query.clone())
            .inference(InferenceMode::Bayesian(crate::BayesianConfig::conjugate().n_draws(24)))
            .refute(RefuteSuite::None);
        let study = builder.build().expect("study build");
        let physical = study.plan(&ctx).expect("physical plan");
        let outcome = match query {
            CausalQuery::AnomalyAttribution(query) => *query.targets.first().unwrap(),
            CausalQuery::ChangeAttribution(query) => query.outcome,
            _ => unreachable!(),
        };
        let (identification, estimand) =
            parametric_scm_identification(query.clone(), outcome, outcome);
        let cache = Arc::new(crate::analysis::prepared::CachedStaticIdentification {
            identification,
            estimand,
        });
        let operation = CheckedAttributionOperation::checked(
            &study,
            &Arc::new(physical.clone()),
            &cache,
            &ExecutionContext::for_tests(0),
        )
        .expect("Bayesian attribution plan");
        let info = operation.info();
        assert_eq!((info.identifier, info.estimator), (IdentifierId::GcmParametric, estimator));
        drop(study);
        operation
    }

    #[test]
    fn anomaly_posterior_executes_from_retained_operation_after_builder_drop() {
        let (data, graph) = anomaly_fixture();
        let query = CausalQuery::AnomalyAttribution(AnomalyAttributionQuery::new(
            [VariableId::from_raw(1)],
            100,
        ));
        let operation = checked_operation(&data, graph, &query, EstimatorId::GcmFitBayesian);
        let result = operation.execute(&data, &context()).expect("retained execution");
        let posterior = result.posterior.as_ref().expect("shared-row posterior");
        assert_eq!(posterior.draws.n_draws, 24);
        assert_eq!(posterior.draws.schema.n_quantities(), 1);
        assert!(posterior.summaries.mean[0].is_finite() && posterior.summaries.mean[0] > 0.0);
        let scores = result.anomaly().expect("anomaly scores");
        let target = scores.iter().find(|scores| scores.target == VariableId::from_raw(1)).unwrap();
        let top_row = target.scores.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        assert_eq!(target.rows[top_row], 19, "known outlier remains the highest-scoring row");
        assert!(result.diagnostics.iter().any(|d| d.code.as_ref() == "gcm.attribution.bayesian"));
    }

    #[test]
    fn change_posterior_keeps_total_and_shapley_components_coupled() {
        let (data, graph) = change_fixture();
        let query = CausalQuery::ChangeAttribution(antecedent_core::ChangeAttributionQuery::new(
            VariableId::from_raw(1),
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
        ));
        let operation =
            checked_operation(&data, graph, &query, EstimatorId::GcmAttributionBayesian);
        let result = operation.execute(&data, &context()).expect("retained execution");
        let posterior = result.posterior.as_ref().expect("shared-population posterior");
        assert_eq!(posterior.draws.n_draws, 24);
        assert!(posterior.summaries.mean[0].is_finite());
        let truth: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../conformance/estimate/staged_attribution/expected.json"
        ))
        .unwrap();
        let expected = truth["change"]["total_change"].as_f64().unwrap();
        let tolerance = truth["change"]["tolerance"].as_f64().unwrap();
        assert!((posterior.summaries.mean[0] - expected).abs() < tolerance);
        for draw in 0..posterior.draws.n_draws {
            let total = posterior.draws.get(draw, 0).unwrap();
            let sum = (1..posterior.draws.schema.n_quantities())
                .map(|quantity| posterior.draws.get(draw, quantity).unwrap())
                .sum::<f64>();
            assert!((sum - total).abs() < 1e-8, "Shapley components must sum to total per draw");
        }
        assert!(result.diagnostics.iter().any(|d| d.code.as_ref() == "gcm.attribution.bayesian"));
    }
}
