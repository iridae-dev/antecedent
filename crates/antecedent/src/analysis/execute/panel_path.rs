// SPDX-License-Identifier: MIT OR Apache-2.0

use super::completion_fit_memo::{CompletionFitMemo, ResponseFitKey};
use super::sequential_validation::SequentialValidation;
use super::{
    AnalyticSeKind, Arc, BayesianConfig, BayesianGCompWorkspace, BayesianGComputationAte,
    BootstrapAccounting, CausalError, CausalPosterior, CausalQuery, CausalResponse, Diagnostic,
    DiagnosticKind, DiagnosticSeverity, DiscoveryEstimationSplit, EffectEstimate,
    EstimationWorkspace, EstimatorId, ExecutionContext, GraphEffectDraws, GraphIdentFlag,
    IdentificationBasis, IdentificationResult, IdentificationStatus, IdentifiedEstimand,
    IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifierId, InferenceMode, Instant,
    ObservationSpec, OverlapPolicy, PanelData, PhysicalExecutionPlan, PredictiveCheckReport,
    PriorSet, RefutationRun, RefuteSuite, ResponseIdentification, ResponseQuery,
    ResponseUncertainty, ResponseValue, Strategy, StudyResult, TableView,
    TemporalBackdoorIdentifier, TemporalDag, TemporalEffectQuery, TemporalIndexer,
    TemporalLinearAdjustment, TemporalRefitContext, TemporalResponseEstimator, UNFOLDED_LINEAR_SEM,
    apply_temporal_prior_sensitivity, bayesian_gcomp, bayesian_temporal_gcomp,
    bootstrap_has_enough_successes, completion_fit_seed, effect_from_posterior, elapsed_ns,
    envelope_se_omits_between_atom_variance, envelope_to_identification_result_for,
    identification_status_ok_for_case, identified_set_interval_diagnostics,
    identify_cached_diagnostic, mag_sequential_unevaluable_diagnostic, mix_weighted_analytic_se,
    nan_effect, posterior_identified_set_interval, provenance_ids, push_conflict_diagnostics,
    report_identify_compute, require_identified, resolve_bayesian_prior_with_conflict,
    same_fit_inputs, same_fitted_mechanisms, same_fitted_problem, select_estimand,
    sequential_linear_sem_assumption, stack_panel_tabular, temporal_refuter_average,
    with_conflict_summary,
};
use crate::planner::RouteDefaults;
use antecedent_core::StreamDomain;

mod class_effect;
mod response;
mod sequential;

use response::panel_unit_average_bootstrap_note;

impl super::Study {
    /// The panel support owner, consulted on the data each execute receives
    /// (a prepared refresh hands in new units).
    pub(super) fn refuse_unsupported_panel(&self, panel: &PanelData) -> Result<(), CausalError> {
        super::super::builder::refuse_unsupported_panel_route(
            &self.query,
            self.graph.class(),
            &self.settings.inference,
            panel,
            self.declared.split.as_ref(),
        )
    }

    pub(super) fn execute_panel(
        &self,
        panel: &PanelData,
        graph: &TemporalDag,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.execute_pooled_panel(panel, graph, query, physical, ctx, PanelClusters::Units)
    }

    pub(super) fn execute_pooled_panel(
        &self,
        panel: &PanelData,
        graph: &TemporalDag,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        clusters: PanelClusters,
    ) -> Result<StudyResult, CausalError> {
        self.refuse_unsupported_panel(panel)?;
        let started = Instant::now();
        let (identification, estimand, indexer, identify_cached) =
            self.identify_pooled_panel(graph, query, Some(physical), ctx)?;
        require_identified(&identification)?;

        let unit_ids = panel_unit_ids(panel);
        let atom = PanelPulseAtom::prepare(
            panel,
            &estimand,
            query,
            &indexer,
            self.declared.split.as_ref(),
            ctx,
            1.0,
        )?;
        let (prep, cluster_ids, _) = atom.stacked(&unit_ids)?;
        let mut diagnostics = vec![pooled_panel_estimand_diagnostic(clusters, unit_ids.len())];

        let design = PooledPanelDesign {
            atom: &atom,
            prep: &prep,
            cluster_ids: &cluster_ids,
            unit_ids: &unit_ids,
        };
        let (estimate, mut posterior, estimate_artifact, estimate_op) = self
            .fit_pooled_panel_estimate(&design, &identification, clusters, &mut diagnostics, ctx)?;

        let temporal_ctx = TemporalRefitContext {
            indexer: &indexer,
            temporal_query: query,
            split: self.declared.split.as_ref(),
            kernel_policy: &ctx.kernel_policy,
            time_index: None,
            panel: Some(panel),
            gaps: false,
        };
        let (mut refutations, na_diagnostics) = self.refute_pooled_panel(
            panel,
            temporal_ctx,
            &estimand,
            &estimate,
            if posterior.is_some() {
                EstimatorId::BayesianTemporalGcomp.as_str()
            } else {
                EstimatorId::TemporalLinearAdjustment.as_str()
            },
            ctx,
        )?;
        diagnostics.extend(na_diagnostics);

        posterior = self.pooled_panel_prior_sensitivity(
            &design,
            identification.status,
            estimate.ate,
            posterior,
            &mut refutations,
            ctx,
        )?;

        if let Some(cs) = posterior.as_ref().and_then(|p| p.conflict_summary.as_ref()) {
            push_conflict_diagnostics(&mut diagnostics, cs);
        }

        let (bootstrap_replicates_ok, cancelled) = (
            estimate.uncertainty.bootstrap.replicates_ok,
            estimate.uncertainty.bootstrap.cancelled,
        );
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand,
                estimate,
                Strategy::new(
                    IdentifierId::TemporalBackdoorUnfolded,
                    if posterior.is_some() {
                        EstimatorId::BayesianTemporalGcomp
                    } else {
                        EstimatorId::TemporalLinearAdjustment
                    },
                ),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .refutations(refutations)
            .wall_time_ns(elapsed_ns(started))
            .bootstrap_replicates_ok(bootstrap_replicates_ok)
            .cancelled(cancelled)
            .extras(IdentifiedExecuteExtras {
                identify_provenance: Some(provenance_ids(
                    "identify.temporal_backdoor",
                    "identify.temporal.backdoor.unfolded",
                )),
                estimate_provenance: Some(provenance_ids(estimate_artifact, estimate_op)),
                posterior,
                diagnostics: Some(diagnostics),
                ..Default::default()
            }),
        ))
    }

    /// Pooled panel estimate under the study's inference mode, with its provenance ids.
    fn fit_pooled_panel_estimate(
        &self,
        design: &PooledPanelDesign<'_>,
        identification: &IdentificationResult,
        clusters: PanelClusters,
        diagnostics: &mut Vec<Diagnostic>,
        ctx: &ExecutionContext,
    ) -> Result<(EffectEstimate, Option<CausalPosterior>, &'static str, &'static str), CausalError>
    {
        Ok(match &self.settings.inference {
            InferenceMode::Bayesian(cfg) => {
                let (estimate, posterior) = fit_pooled_panel_bayesian(
                    cfg,
                    design.prep,
                    design.cluster_ids,
                    identification.status,
                    ctx,
                )?;
                (
                    estimate,
                    Some(posterior),
                    "estimate.bayesian_temporal_gcomp.panel",
                    "estimate.bayesian.temporal.gcomp.panel",
                )
            }
            InferenceMode::Frequentist => {
                let estimate = self.fit_pooled_panel_frequentist(
                    design.atom,
                    design.unit_ids,
                    identification,
                    clusters,
                    diagnostics,
                    ctx,
                )?;
                (
                    estimate,
                    None,
                    "estimate.temporal_linear_adjustment.panel",
                    "estimate.temporal.linear.adjustment.panel",
                )
            }
        })
    }

    /// Prior-sensitivity grid of a Bayesian panel under the Full suite (a no-op otherwise).
    fn pooled_panel_prior_sensitivity(
        &self,
        design: &PooledPanelDesign<'_>,
        status: IdentificationStatus,
        ate: f64,
        posterior: Option<CausalPosterior>,
        refutations: &mut Vec<antecedent_validate::RefutationReport>,
        ctx: &ExecutionContext,
    ) -> Result<Option<CausalPosterior>, CausalError> {
        // Panel Bayesian: α-grid under Full when external compose is present (mirror temporal).
        if matches!(self.settings.refute, RefuteSuite::Full) {
            if let (InferenceMode::Bayesian(cfg), Some(post)) =
                (&self.settings.inference, &posterior)
            {
                let mut bprep = BayesianGComputationAte::from_prepared_estimation(design.prep);
                bprep.unit_ids = Some(design.cluster_ids.to_vec());
                return Ok(Some(apply_temporal_prior_sensitivity(
                    cfg,
                    &bprep,
                    status,
                    post,
                    ate,
                    ctx,
                    refutations,
                )?));
            }
        }
        Ok(posterior)
    }

    /// Identifies the panel's pooled Pulse estimand, or reads the horizon entry the
    /// prepared cache or, given the executing `physical` plan, its lowered program
    /// retained.
    pub(super) fn identify_pooled_panel(
        &self,
        graph: &TemporalDag,
        query: &TemporalEffectQuery,
        physical: Option<&PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<(IdentificationResult, IdentifiedEstimand, TemporalIndexer, bool), CausalError>
    {
        let reused = match physical {
            Some(physical) => self.reuse_temporal_identification(physical),
            None => self.temporal_cache().cloned().map(|cache| (cache, true)),
        };
        if let Some((cache, prepared)) = reused {
            let entry = cache.get(query.horizon_steps).ok_or_else(|| CausalError::Compile {
                message: "prepared panel identification cache missing query horizon".into(),
            })?;
            Ok((
                entry.identification.clone(),
                entry.estimand.clone(),
                entry.indexer.clone(),
                prepared,
            ))
        } else {
            let id_res = TemporalBackdoorIdentifier::new()
                .with_parent_adjustment_fallback()
                .identify_temporal(graph, query)
                .map_err(CausalError::from)?;
            report_identify_compute(ctx);
            let identification = id_res.result;
            require_identified(&identification)?;
            let estimand = select_estimand(&identification, EstimatorId::TemporalLinearAdjustment)?;
            Ok((identification, estimand, id_res.indexer, false))
        }
    }

    /// Pooled frequentist panel fit with its unit bootstrap and cluster-SE disclosures.
    fn fit_pooled_panel_frequentist(
        &self,
        atom: &PanelPulseAtom,
        unit_ids: &[u32],
        identification: &IdentificationResult,
        clusters: PanelClusters,
        diagnostics: &mut Vec<Diagnostic>,
        ctx: &ExecutionContext,
    ) -> Result<EffectEstimate, CausalError> {
        let mut estimate = atom.fit(unit_ids, ctx, identification.required_assumptions.clone())?;
        estimate.provenance.assumptions.push(panel_estimand_assumption(clusters, unit_ids.len()));
        let boot = panel_unit_bootstrap(
            std::slice::from_ref(atom),
            self.settings.bootstrap_replicates,
            ctx,
        );
        boot.attach(&mut estimate, self.settings.bootstrap_replicates);
        diagnostics.push(panel_cluster_se_diagnostic(clusters, unit_ids.len()));
        diagnostics.extend(boot.diagnostic(clusters, self.settings.bootstrap_replicates));
        Ok(estimate)
    }

    /// Runs the refutation suite on the stacked panel table.
    fn refute_pooled_panel(
        &self,
        panel: &PanelData,
        temporal: TemporalRefitContext<'_>,
        estimand: &IdentifiedEstimand,
        estimate: &EffectEstimate,
        estimator: &str,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
        let stacked = stack_panel_tabular(panel).map_err(CausalError::from)?;
        let ate_q = temporal_refuter_average(temporal.temporal_query);
        let mut refute_ws = EstimationWorkspace::default();
        RefutationRun {
            data: &stacked,
            estimand,
            query: &ate_q,
            estimate,
            suite: self.settings.refute,
            estimator,
            custom: &self.declared.custom_validators,
            temporal: Some(temporal),
        }
        .reports(&mut refute_ws, None, ctx)
    }
}
/// Bayesian pooled panel posterior with the resolved prior and its conflict summary.
fn fit_pooled_panel_bayesian(
    cfg: &BayesianConfig,
    prep: &antecedent_estimate::PreparedEstimationProblem,
    cluster_ids: &[u32],
    status: IdentificationStatus,
    ctx: &ExecutionContext,
) -> Result<(EffectEstimate, CausalPosterior), CausalError> {
    let mut bayes = bayesian_temporal_gcomp(cfg, ctx);
    let mut bprep = BayesianGComputationAte::from_prepared_estimation(prep);
    bprep.unit_ids = Some(cluster_ids.to_vec());
    let (resolved_prior, conflict_summary) =
        resolve_bayesian_prior_with_conflict(cfg, &bprep, Some(ctx))?;
    bayes.inner.prior = resolved_prior;
    let mut ws = BayesianGCompWorkspace::default();
    let mut posterior = bayes.fit(&bprep, status, &mut ws, ctx).map_err(CausalError::from)?;
    if let Some(summary) = conflict_summary {
        posterior = with_conflict_summary(posterior, summary);
    }
    let estimate = effect_from_posterior(&posterior)?;
    Ok((estimate, posterior))
}

struct PooledPanelDesign<'a> {
    atom: &'a PanelPulseAtom,
    prep: &'a antecedent_estimate::PreparedEstimationProblem,
    cluster_ids: &'a [u32],
    unit_ids: &'a [u32],
}

/// Fewest lag-aligned rows a unit may contribute to a per-unit panel fit.
///
/// A per-unit route weighs every unit 1/N whatever its length, so a unit too short
/// for its own fit would carry the same weight as a long one. The floor is the
/// effective-row threshold below which a temporal response band is not calibrated
/// ([`antecedent_estimate::RESPONSE_SHORT_SERIES_ROWS`]).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "RESPONSE_SHORT_SERIES_ROWS is the positive integer-valued constant 15.0, so the cast is exact"
)]
const PANEL_MIN_UNIT_ROWS: usize = antecedent_estimate::RESPONSE_SHORT_SERIES_ROWS as usize;

/// Refuse a panel whose unit has fewer than [`PANEL_MIN_UNIT_ROWS`] rows left after
/// the structural window `span` (history plus horizon) is aligned.
fn refuse_short_panel_units(panel: &PanelData, span: usize) -> Result<(), CausalError> {
    for unit in panel.units() {
        let rows = unit.series.row_count().saturating_sub(span);
        if rows < PANEL_MIN_UNIT_ROWS {
            return Err(CausalError::Compile {
                message: format!(
                    "panel unit {} has {rows} lag-aligned rows, below the {PANEL_MIN_UNIT_ROWS} \
                     a per-unit panel fit requires: every unit carries weight 1/N in the \
                     equal-weight average whatever its length",
                    unit.unit_id
                ),
            });
        }
    }
    Ok(())
}

/// Structural window of an indexer: history plus horizon.
fn indexer_span(indexer: &TemporalIndexer) -> usize {
    indexer.history() as usize + indexer.horizon() as usize
}

/// What a pooled panel regression clusters on, and (for the record only) the
/// equal-weight average of per-cluster fits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PanelClusters {
    /// Panel units.
    Units,
    /// Environments of a multi-environment study.
    Environments,
    /// Not a pooled fit: the equal-weight average of per-unit fits.
    UnitAverage,
}

impl PanelClusters {
    const fn label(self) -> &'static str {
        match self {
            Self::Units | Self::UnitAverage => "unit",
            Self::Environments => "environment",
        }
    }
}

/// The estimand a pooled panel regression answers.
fn pooled_panel_estimand_diagnostic(clusters: PanelClusters, units: usize) -> Diagnostic {
    let label = clusters.label();
    Diagnostic::new(
        "estimate.temporal_effect.panel.estimand",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "pooled common-coefficient panel effect over {units} {label}s: one coefficient is \
             fit on the stacked {label} rows, so a {label}'s influence grows with its length \
             and its within-{label} treatment variance. This is not the equal-weight average \
             of {label}-specific effects that the panel response and multi-step Sustained \
             routes report; under {label} heterogeneity the two differ"
        ),
    )
}

/// The estimand an equal-weight average of per-unit fits answers.
fn unit_average_estimand_diagnostic(code: &'static str, units: usize) -> Diagnostic {
    Diagnostic::new(
        code,
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "equal-weight average of {units} unit-specific fits: every unit weighs 1/{units} \
             whatever its length, targeting the mean over the unit population. This is not the \
             pooled common-coefficient effect that panel Pulse reports; under unit \
             heterogeneity the two differ"
        ),
    )
}

/// The estimand record every panel route carries on its estimate.
fn panel_estimand_assumption(
    clusters: PanelClusters,
    units: usize,
) -> antecedent_core::AssumptionRecord {
    let label = clusters.label();
    let (id, description) = if clusters == PanelClusters::UnitAverage {
        (
            "panel.estimand.equal_weight_unit_average",
            format!(
                "the reported effect is the equal-weight average of {units} unit-specific fits \
                 (each unit weighs 1/{units} whatever its length), not the pooled \
                 common-coefficient panel effect"
            ),
        )
    } else {
        (
            "panel.estimand.pooled_common_coefficient",
            format!(
                "the reported effect is the pooled common-coefficient panel effect over {units} \
                 {label}s (one coefficient on the stacked {label} rows; a {label}'s influence \
                 grows with its length and within-{label} treatment variance), not the \
                 equal-weight average of {label}-specific effects"
            ),
        )
    };
    antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::ParametricRestriction(
            antecedent_core::ParametricAssumption {
                id: Arc::from(id),
                description: Arc::from(description),
            },
        ),
        source: antecedent_core::AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("estimate.temporal.panel"),
        },
        scope: antecedent_core::AssumptionScope::Estimation,
        status: antecedent_core::AssumptionStatus::Declared,
    }
}

/// Two-sided level of every published panel interval (the result's interval
/// binding): the level the facade publishes, not a panel-specific one.
const PANEL_INTERVAL_LEVEL: f64 = crate::result::REPORTED_SE_INTERVAL_LEVEL;

/// Estimator for the pooled panel regression: no row bootstrap (the unit cluster
/// bootstrap below resamples whole units) and the declared overlap policy.
pub(super) fn pooled_panel_estimator() -> TemporalLinearAdjustment {
    let mut estimator = TemporalLinearAdjustment::new();
    estimator.inner.inference.bootstrap_replicates = 0;
    estimator.inner.overlap = OverlapPolicy::ExplicitOverride;
    estimator
}

/// One pooled Pulse design (the DAG, or one identified completion): every unit's own
/// lag-aligned block, prepared once and stacked per fit.
pub(super) struct PanelPulseAtom {
    weight: f64,
    units: Vec<antecedent_estimate::PreparedEstimationProblem>,
}

impl PanelPulseAtom {
    pub(super) fn prepare(
        panel: &PanelData,
        estimand: &IdentifiedEstimand,
        query: &TemporalEffectQuery,
        indexer: &TemporalIndexer,
        split: Option<&DiscoveryEstimationSplit>,
        ctx: &ExecutionContext,
        weight: f64,
    ) -> Result<Self, CausalError> {
        let units = pooled_panel_estimator()
            .prepare_panel_units(panel, estimand, query, indexer, split, &ctx.kernel_policy)
            .map_err(CausalError::from)?;
        Ok(Self { weight, units })
    }

    /// Stacked design with the panel's own unit ids.
    pub(super) fn stacked(
        &self,
        unit_ids: &[u32],
    ) -> Result<(antecedent_estimate::PreparedEstimationProblem, Vec<u32>, Vec<i64>), CausalError>
    {
        pooled_panel_estimator()
            .stack_panel_units(&self.units.iter().collect::<Vec<_>>(), unit_ids)
            .map_err(CausalError::from)
    }

    /// Pooled common-coefficient fit with the cluster-by-unit SE
    /// ([`panel_cluster_se_fit`]).
    fn fit(
        &self,
        unit_ids: &[u32],
        ctx: &ExecutionContext,
        assumptions: antecedent_core::AssumptionSet,
    ) -> Result<EffectEstimate, CausalError> {
        let (prep, cluster_ids, panel_times) = self.stacked(unit_ids)?;
        panel_cluster_se_fit(&prep, cluster_ids, panel_times, ctx, assumptions)
    }
}

/// Pooled fit whose analytic SE is the Arellano cluster-by-unit variance at `G − 1`
/// degrees of freedom.
///
/// `PanelClusterHac { lag: 0 }` is the full within-unit meat `Σ_g s_g s_g'`, robust to
/// any within-unit dependence as the number of units grows, with the
/// `G/(G−1)·(n−1)/(n−p)` finite-sample factor. A positive lag would truncate the
/// within-unit covariance to a Bartlett window and under-cover under persistent
/// scores. The SE is then scaled by `t_{G−1}/z` at the 0.95 interval level
/// ([`antecedent_stats::few_cluster_t_ratio`]) so `estimate ± 1.96·se` is the
/// `t_{G−1}` interval the few-cluster reference distribution requires.
fn panel_cluster_se_fit(
    prep: &antecedent_estimate::PreparedEstimationProblem,
    cluster_ids: Vec<u32>,
    panel_times: Vec<i64>,
    ctx: &ExecutionContext,
    assumptions: antecedent_core::AssumptionSet,
) -> Result<EffectEstimate, CausalError> {
    let clusters = distinct_clusters(&cluster_ids);
    let mut estimator = pooled_panel_estimator();
    estimator.inner.inference.cluster_ids = Some(cluster_ids);
    estimator.inner.inference.panel_times = Some(panel_times);
    estimator.inner.inference.se_kind = AnalyticSeKind::PanelClusterHac { lag: 0 };
    let mut workspace = EstimationWorkspace::default();
    let mut estimate =
        estimator.fit(prep, &mut workspace, ctx, assumptions).map_err(CausalError::from)?;
    estimate.uncertainty.analytic.se *=
        antecedent_stats::few_cluster_t_ratio(clusters, PANEL_INTERVAL_LEVEL);
    Ok(estimate)
}

fn distinct_clusters(cluster_ids: &[u32]) -> usize {
    cluster_ids.iter().collect::<std::collections::BTreeSet<_>>().len()
}

/// Panel unit ids in panel order.
pub(super) fn panel_unit_ids(panel: &PanelData) -> Vec<u32> {
    panel.units().iter().map(|unit| unit.unit_id).collect()
}

/// Unit cluster bootstrap of a frozen-weight mixture of pooled Pulse atoms.
struct PanelUnitBootstrap {
    se: Option<f64>,
    completed: u32,
    failed: u32,
    cancelled: bool,
    units: usize,
}

impl PanelUnitBootstrap {
    /// Record the bootstrap on `estimate` (SE, replicate counts, cancellation).
    fn attach(&self, estimate: &mut EffectEstimate, requested: u32) {
        if requested == 0 {
            return;
        }
        estimate.uncertainty.bootstrap.se = self.se;
        estimate.uncertainty.bootstrap.replicates_ok = Some(self.completed);
        estimate.uncertainty.bootstrap.replicates_failed = Some(self.failed);
        estimate.uncertainty.bootstrap.cancelled = self.cancelled;
    }

    fn diagnostic(&self, clusters: PanelClusters, requested: u32) -> Option<Diagnostic> {
        let label = clusters.label();
        (requested > 0).then(|| {
            Diagnostic::new(
                "estimate.temporal_effect.panel.unit_bootstrap",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "{label} cluster bootstrap: {} of {} {label}s resampled with replacement \
                     per replicate, each draw its own cluster, the pooled regression refit on \
                     the restacked {label} blocks (lag windows stay inside their {label}); \
                     {} replicates succeeded, {} failed{}; replicate SD scaled by \
                     sqrt(G/(G-1)) and t_(G-1)/z at the 0.95 level",
                    self.units,
                    self.units,
                    self.completed,
                    self.failed,
                    if self.cancelled { ", cancelled" } else { "" },
                ),
            )
        })
    }
}

/// Resample whole units (with replacement, each draw a fresh cluster), restack each
/// atom's prepared unit blocks, and refit the frozen-weight mixture per replicate.
///
/// Units are prepared once; a replicate only restacks and refits. The replicate SD
/// is scaled by `sqrt(G/(G−1))` (the Arellano finite-sample factor the analytic SE
/// carries) and by `t_{G−1}/z`, so both SEs read on the same `t_{G−1}` scale. The SE is
/// withheld unless at least [`super::PERCENTILE_95_BAND_MIN_SUCCESSES`] replicates
/// succeeded, successes outnumber failures, and the run was not cancelled.
/// Cancellation is not adaptive early-stop: a cancel after the success floor still
/// withholds the SE.
fn panel_unit_bootstrap(
    atoms: &[PanelPulseAtom],
    replicates: u32,
    ctx: &ExecutionContext,
) -> PanelUnitBootstrap {
    let units = atoms.first().map_or(0, |atom| atom.units.len());
    let total_weight: f64 = atoms.iter().map(|atom| atom.weight).sum();
    let mut out = PanelUnitBootstrap { se: None, completed: 0, failed: 0, cancelled: false, units };
    if replicates == 0
        || units < 2
        || !matches!(total_weight.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater))
        || atoms.iter().any(|atom| atom.units.len() != units)
    {
        out.cancelled = ctx.cancellation.is_cancelled();
        return out;
    }
    let estimator = pooled_panel_estimator();
    let fresh_ids: Vec<u32> = (0..u32::try_from(units).unwrap_or(u32::MAX)).collect();
    let mut rng = ctx.rng.stream_for(StreamDomain::Panel, 0xC1A5_5E11);
    let mut workspace = EstimationWorkspace::default();
    // One scalar per replicate. A pool that cannot fund them withholds the bootstrap SE, which
    // is how this helper reports any bootstrap that cannot complete.
    let Ok(_admitted) = ctx.acquire_admission_product(0, u64::from(replicates), 8) else {
        out.cancelled = ctx.cancellation.is_cancelled();
        return out;
    };
    let mut draws = Vec::with_capacity(replicates as usize);
    let mut attempted = 0usize;
    for _ in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            out.cancelled = true;
            break;
        }
        attempted += 1;
        let drawn: Vec<usize> = (0..units)
            .map(|_| usize::try_from(rng.next_u64() % units as u64).unwrap_or(0))
            .collect();
        let mut mixture = 0.0;
        let mut ok = true;
        for atom in atoms {
            let blocks: Vec<_> = drawn.iter().map(|&k| &atom.units[k]).collect();
            let fit = estimator.stack_panel_units(&blocks, &fresh_ids).and_then(|(prep, _, _)| {
                estimator.inner.fit_point(
                    &prep,
                    &mut workspace,
                    antecedent_core::AssumptionSet::default(),
                )
            });
            match fit {
                Ok(estimate) if estimate.ate.is_finite() => {
                    mixture += atom.weight / total_weight * estimate.ate;
                }
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            draws.push(mixture);
        } else {
            out.failed += 1;
        }
    }
    out.completed = u32::try_from(draws.len()).unwrap_or(u32::MAX);
    if !out.cancelled && bootstrap_has_enough_successes(draws.len(), attempted) {
        let g = units as f64;
        let sd = antecedent_stats::sample_std(&draws);
        let se = sd
            * (g / (g - 1.0)).sqrt()
            * antecedent_stats::few_cluster_t_ratio(units, PANEL_INTERVAL_LEVEL);
        out.se = se.is_finite().then_some(se);
    }
    out
}

/// Diagnostic naming the analytic panel cluster SE and its degrees of freedom.
fn panel_cluster_se_diagnostic(clusters: PanelClusters, units: usize) -> Diagnostic {
    let label = clusters.label();
    Diagnostic::new(
        "estimate.temporal_effect.panel.cluster_se",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "analytic SE is the Arellano cluster-by-{label} variance over G={units} {label}s \
             (full \
             within-{label} score covariance, G/(G-1)(n-1)/(n-p) factor) with G-1={} degrees of \
             freedom: it is scaled by t_(G-1)/z = {:.4} so estimate ± 1.96·SE is the 95% \
             t_(G-1) interval",
            units.saturating_sub(1),
            antecedent_stats::few_cluster_t_ratio(units, PANEL_INTERVAL_LEVEL),
        ),
    )
}
