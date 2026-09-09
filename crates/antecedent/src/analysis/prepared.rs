//! Compile-once / re-estimate-many prepared analysis handle.
//!
//! Rediscover policy: structure is frozen at prepare time. Changing bootstrap,
//! prior scale, treatment levels, or latency never re-runs discovery — only an
//! explicit new discover / review → prepare cycle may replace the graph.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;
use std::time::Instant;

use antecedent_core::{
    AverageEffectQuery, CausalQuery, CausalSchema, ExecutionContext, Intervention,
    TargetPopulation, TemporalEffectQuery, TemporalResponseSpec, Value,
};
use antecedent_data::{TableView, TabularData, TemporalIndexer, TimeSeriesData};
use antecedent_discovery::{GraphPosterior, dag_from_adjacency_mask, temporal_dag_from_dbn_masks};
use antecedent_estimate::EstimationWorkspace;

use crate::accepted::GraphClass;
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::planner::PhysicalExecutionPlan;
use crate::result::StudyResult;
use crate::strategy_table::DEFAULT_ESTIMATOR;

use antecedent_expr::IdentifiedEstimand;
use antecedent_graph::{Pag, TemporalDag};
use antecedent_identify::{
    IdentificationEnvelope, IdentificationResult, TemporalBackdoorIdentifier,
    TemporalMediationIdentifier,
};
use antecedent_prob::{GraphIdentFlag, WeightedGraphSamples};

use super::builder::{DataInput, RefuteSuite};
use super::execute::Study;
use super::helpers::{project_for_ate_estimate, run_refuters};
use super::stage::{STAGE_VALIDATE, StageClock};

/// Prepare-time identification products for the static ATE / response path.
///
/// Everything identification reads — identifier, graph, query, RD config — is
/// frozen when the handle is built, and identification is deterministic, so an
/// estimate click reuses these instead of re-running identification. Results
/// carry an `exec.identify.cached` diagnostic so reuse is observable.
#[derive(Clone, Debug)]
pub struct CachedStaticIdentification {
    /// Identification result computed at prepare time.
    pub identification: IdentificationResult,
    /// Estimand selected for the prepared estimator.
    pub estimand: IdentifiedEstimand,
}

/// Prepare-time generalized-adjustment envelope for a supplied PAG.
#[derive(Clone, Debug)]
pub(crate) struct CachedPagIdentification {
    /// Per-completion identification results and their probability mass.
    pub envelope: IdentificationEnvelope<Pag>,
    /// Public aggregate identification result derived from [`Self::envelope`].
    pub identification: IdentificationResult,
}

/// One identified atom in a prepared static graph posterior.
#[derive(Clone, Debug)]
pub(crate) struct CachedGraphPosteriorAtomIdentification {
    /// Opaque graph key retained by the posterior envelope.
    pub key: u64,
    /// Per-atom selected estimand.
    pub estimand: IdentifiedEstimand,
    /// Full per-atom identification result.
    pub identification: IdentificationResult,
}

/// Prepare-time identification for every atom in a static graph posterior.
#[derive(Clone, Debug)]
pub(crate) struct CachedGraphPosteriorIdentification {
    /// Frozen weights, graph keys, and identified/unidentified flags.
    pub graphs: WeightedGraphSamples,
    /// Identified atoms, in posterior order. Unidentified atoms remain in
    /// [`Self::graphs`] with [`GraphIdentFlag::Unidentified`].
    pub atoms: Arc<[CachedGraphPosteriorAtomIdentification]>,
}

/// One identified atom in a prepared DBN graph posterior.
#[derive(Clone, Debug)]
pub(crate) struct CachedDbnPosteriorAtomIdentification {
    /// Collision-free, position-derived key used by the effect envelope.
    ///
    /// [`GraphPosterior::graph_keys`] default to contemporaneous adjacency
    /// masks, which are not unique for DBN atoms that differ only in lagged
    /// edges. The execution key is therefore local to this frozen cache.
    pub key: u64,
    /// Per-atom selected estimand.
    pub estimand: IdentifiedEstimand,
    /// Full temporal per-atom identification result.
    pub identification: IdentificationResult,
    /// Finite-unfolding indexer produced with [`Self::identification`].
    pub indexer: TemporalIndexer,
}

/// Why identification-time DBN atoms were marked unidentified.
///
/// Mass is retained on [`CachedDbnPosteriorIdentification::graphs`]; these
/// counts say *why* so a result does not have to treat unidentified mass as
/// an unexplained residual.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct DbnIdentifyDemotion {
    /// `temporal_dag_from_dbn_masks` rejected the atom.
    pub invalid_graph: usize,
    /// Temporal identification returned an error.
    pub identify_failed: usize,
    /// Identification status is not a licensed identified status.
    pub not_identified: usize,
    /// No selected estimand (empty list or selector refusal).
    pub no_estimand: usize,
}

impl DbnIdentifyDemotion {
    pub(crate) fn total(&self) -> usize {
        self.invalid_graph + self.identify_failed + self.not_identified + self.no_estimand
    }

    pub(crate) fn summary(&self, prepare: usize, fit: usize, draws: usize) -> String {
        let estimate = prepare + fit + draws;
        format!(
            "identify_unidentified={} (invalid_graph={} identify_failed={} not_identified={} no_estimand={}); estimate_demoted={estimate} (prepare={prepare} fit={fit} draws={draws})",
            self.total(),
            self.invalid_graph,
            self.identify_failed,
            self.not_identified,
            self.no_estimand,
        )
    }
}

/// Prepare-time identification for every atom in a DBN graph posterior.
#[derive(Clone, Debug)]
pub(crate) struct CachedDbnPosteriorIdentification {
    /// Frozen weights, graph keys, and identified/unidentified flags.
    pub graphs: WeightedGraphSamples,
    /// Identified atoms, in posterior order. Unidentified atoms remain in
    /// [`Self::graphs`] with [`GraphIdentFlag::Unidentified`].
    pub atoms: Arc<[CachedDbnPosteriorAtomIdentification]>,
    /// Identification-time demotion counts, frozen with the cache.
    pub identify_demotion: DbnIdentifyDemotion,
}

/// Identify every atom in a static graph posterior and retain its original mass.
///
/// This is shared by fresh execution and [`Study::prepare`]. Calling it for a
/// fresh run performs identification; prepared estimate clicks clone the result
/// stored on the handle instead.
pub(crate) fn build_graph_posterior_identification_cache(
    posterior: &GraphPosterior,
    query: &AverageEffectQuery,
    ctx: &ExecutionContext,
) -> Result<CachedGraphPosteriorIdentification, CausalError> {
    use std::collections::HashMap;

    use crate::strategy_table::{
        DEFAULT_IDENTIFIER_ID, EstimatorId, identify_static, select_estimand,
    };

    // A DBN posterior's contemporaneous masks are valid DAGs, but identifying a
    // static effect on them alone would drop every lagged confounder. That is
    // a different coordinate (temporal Pulse / Sustained), so fail closed.
    if posterior.lag_masks.is_some() || posterior.max_lag.is_some() {
        return Err(CausalError::Unsupported {
            message: "static AverageEffect over a graph posterior requires static DAG atoms; \
                      this posterior carries DBN lag structure, so it needs a temporal \
                      Pulse / single-step Sustained query",
        });
    }
    super::execute::report_identify_compute(ctx);
    let mut weights = Vec::with_capacity(posterior.n_graphs);
    let mut flags = Vec::with_capacity(posterior.n_graphs);
    let mut keys = Vec::with_capacity(posterior.n_graphs);
    let mut atoms = Vec::new();
    let mut by_mask: HashMap<u64, Option<(IdentifiedEstimand, IdentificationResult)>> =
        HashMap::new();

    for i in 0..posterior.n_graphs {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
        }
        if let Some(progress) = &ctx.progress {
            #[allow(clippy::cast_precision_loss)]
            progress.report(i as f64 / posterior.n_graphs.max(1) as f64, "envelope.identify");
        }
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
        }
        let mask = posterior.adjacency[i];
        let key = posterior.graph_keys[i];
        keys.push(key);
        weights.push(posterior.weights[i]);
        let resolved = if let Some(hit) = by_mask.get(&mask) {
            hit.clone()
        } else {
            let value =
                (|| -> Result<Option<(IdentifiedEstimand, IdentificationResult)>, CausalError> {
                    let Ok(dag) = dag_from_adjacency_mask(mask, posterior.n_vars) else {
                        return Ok(None);
                    };
                    let Ok(identification) = identify_static(DEFAULT_IDENTIFIER_ID, &dag, query)
                    else {
                        return Ok(None);
                    };
                    if !super::execute::identification_status_ok_for_case(identification.status)
                        || identification.estimands.is_empty()
                    {
                        return Ok(None);
                    }
                    let estimand = select_estimand(&identification, EstimatorId::BayesianGcomp)?;
                    Ok(Some((estimand, identification)))
                })()?;
            by_mask.insert(mask, value.clone());
            value
        };
        if let Some((estimand, identification)) = resolved {
            flags.push(GraphIdentFlag::Identified);
            atoms.push(CachedGraphPosteriorAtomIdentification { key, estimand, identification });
        } else {
            flags.push(GraphIdentFlag::Unidentified);
        }
    }
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
    }

    let graphs = WeightedGraphSamples::new(weights, flags, keys)
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
    Ok(CachedGraphPosteriorIdentification { graphs, atoms: Arc::from(atoms) })
}

/// Identify every atom in a DBN posterior and retain its original mass.
///
/// The temporal indexer is part of the cached identification product: rebuilding
/// it from refreshed data would silently couple identification to the estimate
/// click even though graph and query are frozen.
pub(crate) fn build_dbn_posterior_identification_cache(
    posterior: &GraphPosterior,
    variables: &[antecedent_core::VariableId],
    query: &TemporalEffectQuery,
    ctx: &ExecutionContext,
) -> Result<CachedDbnPosteriorIdentification, CausalError> {
    use crate::strategy_table::{EstimatorId, select_estimand};

    let lag_masks = posterior.lag_masks.as_ref().ok_or_else(|| CausalError::Compile {
        message: "DBN posterior missing per-atom lag masks".into(),
    })?;
    let max_lag = posterior
        .max_lag
        .ok_or_else(|| CausalError::Compile { message: "DBN posterior missing max_lag".into() })?;
    super::execute::report_identify_compute(ctx);
    let mut weights = Vec::with_capacity(posterior.n_graphs);
    let mut flags = Vec::with_capacity(posterior.n_graphs);
    let mut keys = Vec::with_capacity(posterior.n_graphs);
    let mut atoms = Vec::new();
    let mut identify_demotion = DbnIdentifyDemotion::default();

    for i in 0..posterior.n_graphs {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
        }
        if let Some(progress) = &ctx.progress {
            #[allow(clippy::cast_precision_loss)]
            progress.report(i as f64 / posterior.n_graphs.max(1) as f64, "envelope.identify");
        }
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
        }
        // A DBN atom is the pair (contemporaneous mask, lag mask), but the
        // public GraphPosterior constructor keys atoms by contemporaneous mask
        // alone. Use posterior position as a collision-free execution key so
        // lag-distinct atoms keep distinct fits, flags, and weights inside the
        // effect envelope. This key is internal and does not alter the public
        // GraphPosterior representation.
        let key = dbn_envelope_key(i)?;
        keys.push(key);
        weights.push(posterior.weights[i]);
        let Ok(graph) = temporal_dag_from_dbn_masks(
            posterior.adjacency[i],
            lag_masks[i],
            posterior.n_vars,
            max_lag,
            variables,
        ) else {
            flags.push(GraphIdentFlag::Unidentified);
            identify_demotion.invalid_graph += 1;
            continue;
        };
        let Ok(temporal) = TemporalBackdoorIdentifier::new().identify_temporal(&graph, query)
        else {
            flags.push(GraphIdentFlag::Unidentified);
            identify_demotion.identify_failed += 1;
            continue;
        };
        let identification = temporal.result;
        if !super::execute::identification_status_ok_for_case(identification.status) {
            flags.push(GraphIdentFlag::Unidentified);
            identify_demotion.not_identified += 1;
            continue;
        }
        if identification.estimands.is_empty() {
            flags.push(GraphIdentFlag::Unidentified);
            identify_demotion.no_estimand += 1;
            continue;
        }
        let Ok(estimand) = select_estimand(&identification, EstimatorId::TemporalLinearAdjustment)
        else {
            flags.push(GraphIdentFlag::Unidentified);
            identify_demotion.no_estimand += 1;
            continue;
        };
        flags.push(GraphIdentFlag::Identified);
        atoms.push(CachedDbnPosteriorAtomIdentification {
            key,
            estimand,
            identification,
            indexer: temporal.indexer,
        });
    }
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
    }

    let graphs = WeightedGraphSamples::new(weights, flags, keys)
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
    Ok(CachedDbnPosteriorIdentification { graphs, atoms: Arc::from(atoms), identify_demotion })
}

fn dbn_envelope_key(index: usize) -> Result<u64, CausalError> {
    u64::try_from(index).map_err(|_| CausalError::Compile {
        message: "DBN posterior has too many atoms for envelope keys".into(),
    })
}

/// Prepare-time identification products for one temporal horizon.
#[derive(Clone, Debug)]
pub struct CachedTemporalHorizonIdentification {
    /// Requested horizon these products were identified for.
    pub horizon: u32,
    /// Unfolded backdoor identification at [`Self::horizon`].
    pub identification: IdentificationResult,
    /// Estimand selected for this horizon.
    pub estimand: IdentifiedEstimand,
    /// Finite-unfolding indexer paired with [`Self::identification`].
    pub indexer: TemporalIndexer,
}

/// Prepare-time temporal-backdoor identification (ADR 0021).
///
/// For temporal [`CausalQuery::Response`], identification + lag indexer are
/// frozen once per unique requested horizon. For scalar
/// [`CausalQuery::TemporalEffect`] (Pulse / single-step Sustained), they are
/// frozen for that query's horizon. Estimate clicks reuse the cache and must
/// not re-identify. A union of per-horizon adjustment sets is not treated as
/// one shared `Z`.
#[derive(Clone, Debug)]
pub struct CachedTemporalIdentification {
    /// One entry per unique requested horizon, in query order.
    pub by_horizon: Arc<[CachedTemporalHorizonIdentification]>,
}

impl CachedTemporalIdentification {
    /// Identification products for `horizon`, if prepared.
    #[must_use]
    pub fn get(&self, horizon: u32) -> Option<&CachedTemporalHorizonIdentification> {
        self.by_horizon.iter().find(|entry| entry.horizon == horizon)
    }
}

/// Durable handle: fixed schema, graph, query, and estimator; swap data and re-estimate.
///
/// Created via [`Study::prepare`]. Discovery / review-required graphs are refused —
/// prepare is for the interactive estimate click path on an already-accepted artifact.
///
/// **Frozen at prepare:** schema (names, types, order); graph / `AcceptedGraph`
/// or supplied graph-posterior atoms and weights; query identity; identifier;
/// observation / transport / interference assumptions; target-population
/// bindings.
///
/// **Estimate click:** same-schema data; estimator numeric knobs, latency,
/// seeds, bootstrap; `ExecutionContext` budget / cancellation. Does not
/// re-identify or recompile the logical plan.
///
/// **Refute click:** same frozen identification and estimand; schema-gated
/// data and suite. Currently [`CausalQuery::AverageEffect`] only.
///
/// **Re-prepare required:** any frozen field change, including schema
/// mismatch. Changing a frozen field on [`Self::refresh`] is an error, not a
/// silent recompile. `analyze` / [`Study::run`] is sugar over identify →
/// prepare → estimate.
#[derive(Clone, Debug)]
pub struct PreparedStudy {
    /// Frozen analysis config (data slot replaced on each estimate).
    analysis: Study,
    /// Ready physical plan from the prepare-time compile (never recompiled on refresh).
    plan: PhysicalExecutionPlan,
    /// Schema fingerprint from prepare-time data.
    schema: CausalSchema,
    /// Sampling regularity frozen for temporal prepares (`None` = tabular).
    time_regularity: Option<antecedent_data::SamplingRegularity>,
}

impl PreparedStudy {
    /// Borrow the caller's frozen query, with original variable ids and query kind.
    #[must_use]
    pub fn query(&self) -> &CausalQuery {
        &self.analysis.query
    }

    /// Borrow the frozen schema fingerprint.
    #[must_use]
    pub fn schema(&self) -> &CausalSchema {
        &self.schema
    }

    /// Matrix structure-source axis frozen at prepare.
    #[must_use]
    pub const fn structure_source(&self) -> crate::support::StructureSource {
        self.analysis.structure_source()
    }

    /// Evidence contract frozen at prepare. `None` when the query is off-axis.
    #[must_use]
    pub const fn support_status(&self) -> Option<crate::support::CellStatus> {
        self.analysis.support_status()
    }

    /// Borrow the ready physical plan retained from prepare.
    #[must_use]
    pub fn plan(&self) -> &PhysicalExecutionPlan {
        &self.plan
    }

    /// Re-estimate on `data` without recompiling the physical plan.
    ///
    /// # Errors
    ///
    /// Schema incompatibility, identification / estimation / validation failures.
    pub fn estimate(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.ensure_schema_compatible(data)?;
        self.analysis.execute_tabular(data, &self.plan, ctx)
    }

    /// Replace retained data and re-estimate (same semantics as [`Self::estimate`]).
    ///
    /// # Errors
    ///
    /// Schema incompatibility, identification / estimation / validation failures.
    pub fn refresh(
        &mut self,
        data: TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.ensure_schema_compatible(&data)?;
        self.analysis.data = DataInput::Tabular(data);
        self.analysis.execute(&self.plan, ctx)
    }

    /// Second-click / background refute: replace validation on a prior estimate.
    ///
    /// Leaves ATE / identification / estimand unchanged. Records `validate` stage timing.
    /// Prefer `suite=PlaceboAndRcc` or `Full` after an interactive first click with
    /// Cheap / None.
    ///
    /// # Errors
    ///
    /// Schema mismatch, missing AverageEffect query, cancel, or validator failures.
    pub fn refute(
        &self,
        prior: &StudyResult,
        data: &TabularData,
        suite: RefuteSuite,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.ensure_schema_compatible(data)?;
        if let CausalQuery::Mediation(query) = &self.analysis.query {
            if prior.treatment != query.treatment
                || prior.outcome != query.outcome
                || prior.identification.query != self.analysis.query
            {
                return Err(CausalError::Compile {
                    message: "refute prior does not match mediation query".into(),
                });
            }
            let graph = self.analysis.graph.as_dag().ok_or(CausalError::Unsupported {
                message: "static mediation refute requires Dag",
            })?;
            let mediation = prior.mediation.as_ref().ok_or(CausalError::Unsupported {
                message: "refute requires prior mediation result",
            })?;
            let mut result = prior.clone();
            result.refutations = if suite == RefuteSuite::None {
                Vec::new()
            } else {
                antecedent_validate::mediation::refute_static_mediation(
                    data,
                    graph,
                    query,
                    mediation,
                    suite == RefuteSuite::Full,
                    ctx,
                )?
            };
            return Ok(result);
        }
        let CausalQuery::AverageEffect(query) = &self.analysis.query else {
            return Err(CausalError::Support {
                id: crate::support::SupportRefusal::Refused,
                message: "PreparedStudy::refute is licensed for AverageEffect only",
            });
        };
        if prior.treatment != query.treatment || prior.outcome != query.outcome {
            return Err(CausalError::Compile {
                message: "refute prior result treatment/outcome does not match prepared query"
                    .into(),
            });
        }
        if matches!(self.analysis.inference, InferenceMode::Bayesian(_)) {
            // Bayesian validation also includes prior/posterior predictive checks
            // (and, for Full, prior sensitivity/MCMC diagnostics). Re-running the
            // frozen physical plan is the only path that constructs those artifacts;
            // retain the prior point/identification while replacing validation state.
            let started = Instant::now();
            let mut analysis = self.analysis.clone();
            analysis.refute = suite;
            let validated =
                analysis.execute_on(&DataInput::Tabular(data.clone()), &self.plan, ctx)?;
            let validate_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let mut out = prior.clone();
            out.refutations = validated.refutations;
            out.predictive_checks = validated.predictive_checks;
            out.posterior = validated.posterior;
            out.performance.stage_timings_ns.push((Arc::from(STAGE_VALIDATE), validate_ns));
            out.performance.wall_time_ns =
                Some(out.performance.wall_time_ns.unwrap_or(0).saturating_add(validate_ns));
            out.diagnostics.push(antecedent_core::Diagnostic::new(
                "exec.refute.second_click",
                antecedent_core::DiagnosticKind::Execution,
                antecedent_core::DiagnosticSeverity::Info,
                format!("second-click refute suite={}", suite.diagnostic_label()),
            ));
            return Ok(out);
        }
        let estimator = self.plan.logical.record.estimator.as_deref().unwrap_or(DEFAULT_ESTIMATOR);

        let (data_est, query_est, estimand_est) =
            project_for_ate_estimate(data, query, &prior.estimand)?;

        let mut clock = StageClock::new();
        clock.begin(ctx, STAGE_VALIDATE, 0.8)?;
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: STAGE_VALIDATE });
        }
        let mut workspace = EstimationWorkspace::default();
        let started = Instant::now();
        let (reports, na_diagnostics) = run_refuters(
            &data_est,
            &estimand_est,
            &query_est,
            &prior.estimate,
            &mut workspace,
            None,
            ctx,
            suite,
            estimator,
            &self.analysis.custom_validators,
            None,
        )?;
        clock.finish(STAGE_VALIDATE);
        let validate_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);

        let mut out = prior.clone();
        out.refutations = reports;
        out.diagnostics.extend(na_diagnostics);
        out.performance.stage_timings_ns.push((Arc::from(STAGE_VALIDATE), validate_ns));
        out.performance.wall_time_ns =
            Some(out.performance.wall_time_ns.unwrap_or(0).saturating_add(validate_ns));
        let suite_label: Arc<str> = Arc::from(suite.diagnostic_label());
        out.diagnostics.push(antecedent_core::Diagnostic::new(
            "exec.refute.second_click",
            antecedent_core::DiagnosticKind::Execution,
            antecedent_core::DiagnosticSeverity::Info,
            format!("second-click refute suite={suite_label}"),
        ));
        let _ = clock.wall_time_ns();
        Ok(out)
    }

    fn ensure_schema_compatible(&self, data: &TabularData) -> Result<(), CausalError> {
        if self.time_regularity.is_some() {
            return Err(CausalError::Compile {
                message: "prepared temporal analysis requires series data; use estimate_series"
                    .into(),
            });
        }
        if data.schema() != &self.schema {
            return Err(CausalError::Compile {
                message: "prepared analysis refresh requires the same schema \
                    (variable names, types, and order) as prepare-time data"
                    .into(),
            });
        }
        Ok(())
    }

    fn ensure_series_compatible(&self, data: &TimeSeriesData) -> Result<(), CausalError> {
        let Some(expected) = &self.time_regularity else {
            return Err(CausalError::Compile {
                message: "prepared tabular analysis requires tabular data; use estimate".into(),
            });
        };
        if data.schema() != &self.schema {
            return Err(CausalError::Compile {
                message: "prepared temporal analysis requires the same schema \
                    (variable names, types, and order) as prepare-time data"
                    .into(),
            });
        }
        if &data.time_index().regularity != expected {
            return Err(CausalError::Compile {
                message: "prepared temporal analysis requires the same time-index \
                    regularity as prepare-time data; re-prepare after a time-index change"
                    .into(),
            });
        }
        Ok(())
    }

    /// Re-estimate a prepared temporal response on series data (no re-identify).
    ///
    /// # Errors
    ///
    /// Schema / time-index mismatch, or estimation failures.
    pub fn estimate_series(
        &self,
        data: &TimeSeriesData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.ensure_series_compatible(data)?;
        self.analysis.execute_on(&DataInput::Temporal(data.clone()), &self.plan, ctx)
    }

    /// Replace retained series and re-estimate.
    ///
    /// # Errors
    ///
    /// Schema / time-index mismatch, or estimation failures.
    pub fn refresh_series(
        &mut self,
        data: TimeSeriesData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.ensure_series_compatible(&data)?;
        self.analysis.data = DataInput::Temporal(data);
        self.analysis.execute(&self.plan, ctx)
    }
}

impl Study {
    /// Compile once into a durable [`PreparedStudy`] for re-estimate-many.
    ///
    /// Supports:
    /// - tabular [`CausalQuery::AverageEffect`] on a supplied static graph
    /// - tabular [`CausalQuery::AverageEffect`] on a supplied DAG graph posterior
    /// - tabular [`CausalQuery::Response`] on a supplied [`GraphClass::Dag`]
    /// - series temporal [`CausalQuery::Response`] on a supplied [`GraphClass::TemporalDag`]
    /// - series [`CausalQuery::TemporalEffect`] (Pulse / single-step Sustained)
    ///   on a supplied [`GraphClass::TemporalDag`]
    /// - series [`CausalQuery::TemporalEffect`] (Pulse / single-step Sustained)
    ///   on a supplied DBN graph posterior
    /// - series [`CausalQuery::Mediation`] (`TemporalMediationEffect`) on a
    ///   supplied [`GraphClass::TemporalDag`] (static-style cache, not `I(h)`)
    ///
    /// Discovery inputs and review-required compiles are refused.
    ///
    /// # Errors
    ///
    /// Unsupported combination, compile failure, or review-required plan.
    pub fn prepare(&self, ctx: &ExecutionContext) -> Result<PreparedStudy, CausalError> {
        ensure_prepared_supported(self)?;
        let plan = self.compile(ctx)?;
        let (schema, time_regularity) = match &self.data {
            DataInput::Tabular(data) => (data.schema().clone(), None),
            DataInput::Temporal(data) | DataInput::Event(data) => {
                (data.schema().clone(), Some(data.time_index().regularity.clone()))
            }
            _ => {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy requires tabular or temporal series data",
                });
            }
        };
        let mut analysis = self.clone();
        match (&self.data, &self.query, self.graph_posterior.as_ref()) {
            (DataInput::Tabular(_), CausalQuery::AverageEffect(query), Some(posterior)) => {
                analysis.graph_posterior_identification_cache = Some(Arc::new(
                    build_graph_posterior_identification_cache(posterior, query, ctx)?,
                ));
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(query),
                Some(posterior),
            ) => {
                let variables: Vec<_> =
                    data.schema().variables().iter().map(|variable| variable.id).collect();
                analysis.dbn_posterior_identification_cache = Some(Arc::new(
                    build_dbn_posterior_identification_cache(posterior, &variables, query, ctx)?,
                ));
            }
            (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::Mediation(query), None) => {
                analysis.identification_cache =
                    self.prepare_temporal_mediation_identification()?.map(Arc::new);
                if let Some(graph) = self.graph.as_temporal_dag() {
                    analysis.mediation_adjustment_cache =
                        Some(self.mediation_adjustment(graph, query)?);
                }
            }
            (DataInput::Tabular(_), _, None) => {
                analysis.identification_cache =
                    self.prepare_static_identification(&plan)?.map(Arc::new);
                analysis.pag_identification_cache =
                    self.prepare_pag_identification(&plan)?.map(Arc::new);
            }
            (_, _, None) => {
                analysis.temporal_identification_cache =
                    self.prepare_temporal_identification()?.map(Arc::new);
            }
            (_, _, Some(_)) => {
                // `ensure_prepared_supported` already refuses this coordinate; a
                // library must still answer with a typed refusal, never a panic.
                return Err(CausalError::Support {
                    id: crate::support::SupportRefusal::Refused,
                    message: "graph_posterior on the prepared handle is licensed only for \
                        tabular AverageEffect and series TemporalEffect (Pulse / single-step \
                        Sustained)",
                });
            }
        }
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
        }
        // The posterior builders report `identify.compute` themselves; the
        // single-graph, PAG, and temporal caches identify above without a
        // progress hook, so report once here when one of them was built. A
        // sharp-RD prepare builds no cache and reports nothing: its clicks do.
        if analysis.identification_cache.is_some()
            || analysis.pag_identification_cache.is_some()
            || analysis.temporal_identification_cache.is_some()
        {
            super::execute::report_identify_compute(ctx);
        }
        Ok(PreparedStudy { analysis, plan, schema, time_regularity })
    }

    /// Compute the static-path identification once at prepare time.
    ///
    /// Mirrors `execute_static`'s (and, for `bayesian.gcomp`, `execute_bayesian`'s)
    /// stage-1 inputs exactly. Sharp RD and PAG envelopes dispatch to separate
    /// preparation paths. Graph posteriors use their per-atom caches.
    fn prepare_static_identification(
        &self,
        plan: &PhysicalExecutionPlan,
    ) -> Result<Option<CachedStaticIdentification>, CausalError> {
        use crate::strategy_table::{
            DEFAULT_IDENTIFIER, EstimatorId, IdentifierId, identify_static, identify_static_query,
            identify_static_query_with_rd, select_estimand,
        };
        if matches!(self.query, CausalQuery::Counterfactual(_)) {
            let graph = self
                .graph
                .as_dag()
                .ok_or(CausalError::Unsupported { message: "counterfactual requires Dag" })?;
            let identification =
                identify_static_query(IdentifierId::GcmParametric, graph, &self.query)?;
            let estimand = identification.estimands[0].clone();
            return Ok(Some(CachedStaticIdentification { identification, estimand }));
        }
        let identifier = plan.logical.record.identifier.as_deref().unwrap_or(DEFAULT_IDENTIFIER);
        let estimator = plan.logical.record.estimator.as_deref().unwrap_or(DEFAULT_ESTIMATOR);
        let identifier_id: IdentifierId = identifier.parse()?;
        let estimator_id: EstimatorId = estimator.parse()?;
        if matches!(estimator_id, EstimatorId::RdSharp) {
            return Ok(None);
        }
        match &self.query {
            CausalQuery::AverageEffect(query) => {
                if self.graph.class() == GraphClass::Admg
                    && self.graph.as_admg().is_some_and(super::execute::admg_has_bidirected)
                {
                    use crate::strategy_table::identify_admg;

                    let admg = self.graph.as_admg().ok_or_else(|| CausalError::Compile {
                        message: "ADMG prepare missing supplied graph".into(),
                    })?;
                    let identification = identify_admg(identifier_id, admg, query)?;
                    let estimand = select_estimand(&identification, estimator_id)?;
                    return Ok(Some(CachedStaticIdentification { identification, estimand }));
                }
                let graph = match self.graph.class() {
                    GraphClass::Dag => self.graph.as_dag().cloned(),
                    GraphClass::Cpdag | GraphClass::Admg => plan.static_graph().cloned(),
                    _ => None,
                };
                let Some(graph) = graph else {
                    return Ok(None);
                };
                // `execute_bayesian` never consults `self.rd` (it calls
                // `identify_static`, which is `identify_static_query_with_rd`
                // with `rd: None`); mirror that exactly so the cached
                // identification matches what an uncached Bayesian run would
                // compute, even if a caller set `.rd_config(..)` alongside
                // `bayesian.gcomp`.
                let rd = if matches!(estimator_id, EstimatorId::BayesianGcomp) {
                    None
                } else {
                    self.rd.map(|c| {
                        antecedent_identify::SharpRdConfig::new(
                            c.running_variable,
                            c.cutoff,
                            c.bandwidth,
                        )
                    })
                };
                let identification = identify_static_query_with_rd(
                    identifier_id,
                    &graph,
                    &CausalQuery::AverageEffect(query.clone()),
                    rd,
                )?;
                let estimand = select_estimand(&identification, estimator_id)?;
                Ok(Some(CachedStaticIdentification { identification, estimand }))
            }
            CausalQuery::Response(query) => {
                let Some(graph) = self.graph.as_dag().cloned() else {
                    return Ok(None);
                };
                let identification = identify_static_query(
                    identifier_id,
                    &graph,
                    &CausalQuery::Response(query.clone()),
                )?;
                let estimand = identification.estimands.first().cloned().ok_or_else(|| {
                    CausalError::Compile {
                        message: "response identifier returned no estimand".into(),
                    }
                })?;
                Ok(Some(CachedStaticIdentification { identification, estimand }))
            }
            CausalQuery::Mediation(query) if self.graph.class() == GraphClass::Dag => {
                let graph = self.graph.as_dag().expect("Dag");
                let identification = identify_static_query(
                    IdentifierId::PathSpecificNatural,
                    graph,
                    &CausalQuery::Mediation(query.clone()),
                )?;
                let estimand =
                    select_estimand(&identification, EstimatorId::StaticMediationLinear)?;
                Ok(Some(CachedStaticIdentification { identification, estimand }))
            }
            CausalQuery::ConditionalEffect(query) => {
                // Mirrors execute_conditional / execute_bayesian: identifier is
                // builder-selected or defaults to backdoor; estimator follows inference.
                use crate::strategy_table::DEFAULT_CONDITIONAL_IDENTIFIER;
                let Some(graph) = self.graph.as_dag().cloned() else {
                    return Ok(None);
                };
                let identifier = plan
                    .logical
                    .record
                    .identifier
                    .as_deref()
                    .unwrap_or(DEFAULT_CONDITIONAL_IDENTIFIER);
                let identifier_id: IdentifierId = identifier.parse()?;
                let identification = identify_static(identifier_id, &graph, &query.inner)?;
                let estimator_id = if matches!(self.inference, InferenceMode::Bayesian(_)) {
                    EstimatorId::BayesianConditional
                } else {
                    EstimatorId::ConditionalLinearAdjustment
                };
                let estimand = select_estimand(&identification, estimator_id)?;
                Ok(Some(CachedStaticIdentification { identification, estimand }))
            }
            CausalQuery::PathSpecific(query) => {
                // Mirrors `execute_path_specific`'s identify+select-estimand step exactly.
                use crate::strategy_table::{DEFAULT_PATH_ESTIMATOR, DEFAULT_PATH_IDENTIFIER};
                let Some(graph) = self.graph.as_dag().cloned() else {
                    return Ok(None);
                };
                let identifier =
                    plan.logical.record.identifier.as_deref().unwrap_or(DEFAULT_PATH_IDENTIFIER);
                let estimator =
                    plan.logical.record.estimator.as_deref().unwrap_or(DEFAULT_PATH_ESTIMATOR);
                let identifier_id: IdentifierId = identifier.parse()?;
                let estimator_id: EstimatorId = estimator.parse()?;
                let cq = CausalQuery::PathSpecific(query.clone());
                let identification = identify_static_query(identifier_id, &graph, &cq)?;
                let estimand = select_estimand(&identification, estimator_id)?;
                Ok(Some(CachedStaticIdentification { identification, estimand }))
            }
            CausalQuery::Distribution(query) => {
                // Mirrors `execute_distribution`'s identify+select-estimand step exactly.
                use crate::strategy_table::{
                    DEFAULT_DISTRIBUTION_ESTIMATOR, DEFAULT_DISTRIBUTION_IDENTIFIER,
                };
                let Some(graph) = self.graph.as_dag().cloned() else {
                    return Ok(None);
                };
                let identifier = plan
                    .logical
                    .record
                    .identifier
                    .as_deref()
                    .unwrap_or(DEFAULT_DISTRIBUTION_IDENTIFIER);
                let estimator = plan
                    .logical
                    .record
                    .estimator
                    .as_deref()
                    .unwrap_or(DEFAULT_DISTRIBUTION_ESTIMATOR);
                let identifier_id: IdentifierId = identifier.parse()?;
                let estimator_id: EstimatorId = estimator.parse()?;
                let cq = CausalQuery::Distribution(query.clone());
                let identification = identify_static_query(identifier_id, &graph, &cq)?;
                let estimand = select_estimand(&identification, estimator_id)?;
                Ok(Some(CachedStaticIdentification { identification, estimand }))
            }
            _ => Ok(None),
        }
    }

    /// Compute the generalized-adjustment envelope once for a supplied PAG.
    fn prepare_pag_identification(
        &self,
        plan: &PhysicalExecutionPlan,
    ) -> Result<Option<CachedPagIdentification>, CausalError> {
        use crate::strategy_table::{DEFAULT_PAG_IDENTIFIER, IdentifierId, identify_pag};

        let CausalQuery::AverageEffect(query) = &self.query else {
            return Ok(None);
        };
        if self.graph.class() != GraphClass::Pag {
            return Ok(None);
        }
        let pag = plan.static_pag().ok_or_else(|| CausalError::Compile {
            message: "PAG prepare missing resolved static PAG".into(),
        })?;
        let identifier =
            plan.logical.record.identifier.as_deref().unwrap_or(DEFAULT_PAG_IDENTIFIER);
        let identifier_id: IdentifierId = identifier.parse()?;
        let envelope = identify_pag(identifier_id, pag, query)?;
        let identification = super::execute::envelope_to_identification_result(&envelope, query);
        Ok(Some(CachedPagIdentification { envelope, identification }))
    }

    /// Identify once per requested horizon at prepare for temporal response or TemporalEffect.
    fn prepare_temporal_identification(
        &self,
    ) -> Result<Option<CachedTemporalIdentification>, CausalError> {
        use crate::strategy_table::{EstimatorId, select_estimand};
        if self.graph.class() != GraphClass::TemporalDag {
            return Ok(None);
        }
        let graph = self.graph.as_temporal_dag().ok_or_else(|| CausalError::Compile {
            message: "temporal prepare requires TemporalDag".into(),
        })?;
        match &self.query {
            CausalQuery::Response(query) => {
                let Some(temporal) = query.temporal.as_ref() else {
                    return Ok(None);
                };
                let (treatment, outcome) =
                    query.functional.primary_pair().ok_or_else(|| CausalError::Compile {
                        message: "response query has no treatment/outcome pair".into(),
                    })?;
                Ok(Some(identify_temporal_response_horizons(
                    graph,
                    treatment,
                    outcome,
                    temporal,
                    &query.target_population,
                    if matches!(self.inference, InferenceMode::Bayesian(_)) {
                        EstimatorId::TemporalResponseBayesian
                    } else {
                        EstimatorId::TemporalResponseGcomp
                    },
                )?))
            }
            CausalQuery::TemporalEffect(query) => {
                let id_res = TemporalBackdoorIdentifier::new()
                    .identify_temporal(graph, query)
                    .map_err(CausalError::from)?;
                let estimand = select_estimand(
                    &id_res.result,
                    if matches!(query.policy, antecedent_core::TemporalPolicy::Sustained { from, until } if from != until)
                    {
                        EstimatorId::TemporalSequentialGcomp
                    } else {
                        EstimatorId::TemporalLinearAdjustment
                    },
                )?;
                Ok(Some(CachedTemporalIdentification {
                    by_horizon: Arc::from([CachedTemporalHorizonIdentification {
                        horizon: query.horizon_steps,
                        identification: id_res.result,
                        estimand,
                        indexer: id_res.indexer,
                    }]),
                }))
            }
            _ => Ok(None),
        }
    }

    /// Single-shot temporal mediation identification (not per-horizon `I(h)`).
    ///
    /// Mirrors `execute_temporal_mediation`'s identify+select step so a prepared
    /// click reuses [`CachedStaticIdentification`] the same way ConditionalEffect does.
    fn prepare_temporal_mediation_identification(
        &self,
    ) -> Result<Option<CachedStaticIdentification>, CausalError> {
        use crate::strategy_table::{EstimatorId, select_estimand};
        let CausalQuery::Mediation(query) = &self.query else {
            return Ok(None);
        };
        if self.graph.class() != GraphClass::TemporalDag {
            return Ok(None);
        }
        let graph = self.graph.as_temporal_dag().ok_or_else(|| CausalError::Compile {
            message: "temporal mediation prepare requires TemporalDag".into(),
        })?;
        let identification = TemporalMediationIdentifier {
            allow_natural_controlled_alias: true,
            ..TemporalMediationIdentifier::new()
        }
        .identify(graph, query)
        .map_err(CausalError::from)?;
        let estimator_id = if matches!(self.inference, InferenceMode::Bayesian(_)) {
            EstimatorId::BayesianTemporalMediation
        } else {
            EstimatorId::TemporalMediation
        };
        let estimand = select_estimand(&identification, estimator_id)?;
        Ok(Some(CachedStaticIdentification { identification, estimand }))
    }
}

pub(crate) fn identify_temporal_response_horizons(
    graph: &TemporalDag,
    treatment: antecedent_core::VariableId,
    outcome: antecedent_core::VariableId,
    temporal: &TemporalResponseSpec,
    target_population: &TargetPopulation,
    estimator_id: crate::strategy_table::EstimatorId,
) -> Result<CachedTemporalIdentification, CausalError> {
    use crate::strategy_table::select_estimand;
    if temporal.horizons.is_empty() {
        return Err(CausalError::Compile {
            message: "temporal response requires at least one horizon".into(),
        });
    }
    let mut by_horizon = Vec::with_capacity(temporal.horizons.len());
    for &horizon in temporal.horizons.iter() {
        let id_query = TemporalEffectQuery {
            treatment,
            outcome,
            policy: temporal.policy.clone(),
            control: Intervention::set(treatment, Value::f64(0.0)),
            active: Intervention::set(treatment, Value::f64(1.0)),
            horizon_steps: horizon,
            max_history_lag: temporal.max_history_lag,
            target_population: target_population.clone(),
        };
        let id_res = TemporalBackdoorIdentifier::new()
            .identify_temporal(graph, &id_query)
            .map_err(CausalError::from)?;
        let estimand = select_estimand(&id_res.result, estimator_id)?;
        by_horizon.push(CachedTemporalHorizonIdentification {
            horizon,
            identification: id_res.result,
            estimand,
            indexer: id_res.indexer,
        });
    }
    Ok(CachedTemporalIdentification { by_horizon: Arc::from(by_horizon) })
}

fn ensure_prepared_supported(analysis: &Study) -> Result<(), CausalError> {
    if analysis.graph_posterior.is_some() {
        return match (&analysis.data, &analysis.query) {
            (DataInput::Tabular(_), CausalQuery::AverageEffect(_))
            | (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::TemporalEffect(_)) => {
                Ok(())
            }
            _ => Err(CausalError::Support {
                id: crate::support::SupportRefusal::Refused,
                message: "graph_posterior on the prepared handle is licensed only for \
                    tabular AverageEffect and series TemporalEffect (Pulse / single-step \
                    Sustained)",
            }),
        };
    }
    match (&analysis.data, &analysis.query) {
        (DataInput::Tabular(_), CausalQuery::AverageEffect(_)) => {
            if !is_supplied_static_graph(analysis.graph.class()) {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy requires a static Dag/Cpdag/Pag/Admg structure \
                (temporal classes are not session-refreshable here)",
                });
            }
        }
        (DataInput::Tabular(_), CausalQuery::Response(q)) if !q.is_temporal() => {
            if analysis.graph.class() != GraphClass::Dag {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy supports ResponseCurve only on a supplied Dag",
                });
            }
        }
        (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::Response(q))
            if q.is_temporal() =>
        {
            if analysis.graph.class() != GraphClass::TemporalDag {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy supports temporal ResponseCurve only on TemporalDag",
                });
            }
        }
        (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::TemporalEffect(_)) => {
            if analysis.graph.class() != GraphClass::TemporalDag {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy supports TemporalEffect only on TemporalDag",
                });
            }
        }
        (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::Mediation(_)) => {
            if analysis.graph.class() != GraphClass::TemporalDag {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy supports TemporalMediationEffect only on TemporalDag",
                });
            }
        }
        (DataInput::Tabular(_), CausalQuery::Counterfactual(_)) => {
            if analysis.graph.class() != GraphClass::Dag {
                return Err(CausalError::Unsupported { message: "counterfactual requires Dag" });
            }
        }
        (DataInput::Tabular(_), CausalQuery::Mediation(_)) => {
            if analysis.graph.class() != GraphClass::Dag {
                return Err(CausalError::Unsupported { message: "static mediation requires Dag" });
            }
        }
        (DataInput::Tabular(_), CausalQuery::ConditionalEffect(_)) => {
            if analysis.graph.class() != GraphClass::Dag {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy supports ConditionalEffect only on a supplied Dag",
                });
            }
        }
        (DataInput::Tabular(_), CausalQuery::PathSpecific(_)) => {
            if analysis.graph.class() != GraphClass::Dag {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy supports PathSpecific only on a supplied Dag",
                });
            }
        }
        (DataInput::Tabular(_), CausalQuery::Distribution(_)) => {
            if analysis.graph.class() != GraphClass::Dag {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy supports Distribution only on a supplied Dag",
                });
            }
        }
        _ => {
            return Err(CausalError::Unsupported {
                message: "PreparedStudy currently supports AverageEffect, ResponseCurve, \
                    ConditionalEffect, PathSpecific, Distribution, temporal ResponseCurve, \
                    TemporalEffect (Pulse / single-step Sustained), or TemporalMediationEffect",
            });
        }
    }
    Ok(())
}

fn is_supplied_static_graph(class: GraphClass) -> bool {
    matches!(class, GraphClass::Dag | GraphClass::Cpdag | GraphClass::Pag | GraphClass::Admg)
}

impl PreparedStudy {
    /// Control intervention level frozen on a counterfactual ITE query.
    #[must_use]
    pub fn counterfactual_control_level(&self) -> f64 {
        match self.query() {
            CausalQuery::Counterfactual(q) => match &q.control {
                Intervention::Set { value, .. } => value.as_f64().unwrap_or(0.0),
                _ => 0.0,
            },
            _ => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::is_supplied_static_graph;
    use crate::accepted::GraphClass;

    #[test]
    fn supplied_static_graphs_only() {
        assert!(is_supplied_static_graph(GraphClass::Dag));
        assert!(is_supplied_static_graph(GraphClass::Cpdag));
        assert!(is_supplied_static_graph(GraphClass::Pag));
        assert!(is_supplied_static_graph(GraphClass::Admg));
        assert!(!is_supplied_static_graph(GraphClass::TemporalDag));
        assert!(!is_supplied_static_graph(GraphClass::TemporalCpdag));
        assert!(!is_supplied_static_graph(GraphClass::TemporalPag));
    }
}
