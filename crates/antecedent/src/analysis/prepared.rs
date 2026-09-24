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
    AverageEffectQuery, CausalQuery, CausalSchema, ExecutionContext, Intervention, MediationQuery,
    OutcomeFunctional, ResponseQuery, TargetPopulation, TemporalEffectQuery, TemporalResponseSpec,
    Value,
};
use antecedent_data::{PanelData, TableView, TabularData, TemporalIndexer, TimeSeriesData};
use antecedent_discovery::{
    GraphPosterior, dag_from_adjacency_mask, temporal_cpdag_from_dbn_masks,
    temporal_dag_from_dbn_masks, temporal_pag_from_dbn_masks,
};
use antecedent_estimate::{
    AipwAte, CellSaturatedAipw, EffectEstimate, EstimationWorkspace, OverlapPolicy, RetargetResult,
    ScoreTable, crossfit_binary_scores, exceedance_cdf_values,
};

use crate::accepted::GraphClass;
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::planner::PhysicalExecutionPlan;
use crate::result::StudyResult;
use crate::strategy_table::DEFAULT_ESTIMATOR;

use antecedent_expr::IdentifiedEstimand;
use antecedent_graph::{Pag, TemporalDag};
use antecedent_identify::{
    IdentificationEnvelope, IdentificationResult, IdentificationStatus, TemporalBackdoorIdentifier,
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

/// Prepare-time MEC envelope for a supplied CPDAG.
#[derive(Clone, Debug)]
pub(crate) struct CachedCpdagIdentification {
    /// Per-completion identification results and their probability mass.
    pub envelope: IdentificationEnvelope<antecedent_graph::Dag>,
    /// Public aggregate identification result derived from [`Self::envelope`].
    pub identification: IdentificationResult,
}

/// Prepare-time TemporalCpdag/Pag envelope (completions + unfold indexers).
#[derive(Clone, Debug)]
pub(crate) struct CachedTemporalClassIdentification {
    /// Generalized-adjustment envelope over TemporalDag completions.
    pub envelope: antecedent_identify::TemporalClassEnvelope,
    /// Functional-specific certificates for every requested response or mediation horizon.
    pub by_horizon: Vec<(u32, antecedent_identify::TemporalClassEnvelope)>,
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

/// One completion case inside a CPDAG/PAG posterior atom.
#[derive(Clone, Debug)]
pub(crate) struct CachedClassPosteriorCase {
    /// Enumeration weight of this completion.
    pub weight: f64,
    /// Identification status of the completion.
    pub status: IdentificationStatus,
    /// Selected estimand when the completion is estimable.
    pub estimand: Option<IdentifiedEstimand>,
}

/// Class-envelope identification for one CPDAG/PAG posterior atom.
#[derive(Clone, Debug)]
pub(crate) struct CachedClassPosteriorAtomIdentification {
    /// Opaque graph key retained by the posterior envelope.
    pub key: u64,
    /// Envelope-level identification (status, diagnostics, assumptions).
    pub identification: IdentificationResult,
    /// Shared estimand when every identified completion agrees.
    pub invariant: Option<IdentifiedEstimand>,
    /// Per-completion cases, including unidentified completions.
    pub cases: Arc<[CachedClassPosteriorCase]>,
    /// Envelope identified weight before posterior mixing.
    pub identified_weight: f64,
    /// Completions whose search was capped.
    pub truncated_completions: usize,
}

/// Prepare-time identification for every atom in a static graph posterior.
#[derive(Clone, Debug)]
pub(crate) struct CachedGraphPosteriorIdentification {
    /// Frozen weights, graph keys, and identified/unidentified flags.
    pub graphs: WeightedGraphSamples,
    /// Identified DAG atoms, one per distinct graph key, in order of first
    /// appearance. Weight an atom by the combined identified mass of its key in
    /// [`Self::graphs`] (`identified_weight_for_key`), which keeps one entry per
    /// posterior sample. Unidentified atoms remain in [`Self::graphs`] with
    /// [`GraphIdentFlag::Unidentified`].
    pub atoms: Arc<[CachedGraphPosteriorAtomIdentification]>,
    /// CPDAG/PAG posterior atoms evaluated with the class ATE envelope.
    /// Empty when [`antecedent_discovery::GraphPosterior::atom_kind`] is DAG or ADMG.
    pub class_atoms: Arc<[CachedClassPosteriorAtomIdentification]>,
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
    /// Per-horizon `I(h)` for a mediation or temporal-response atom. Contrast
    /// atoms leave this empty and use [`Self::identification`] / [`Self::indexer`]
    /// for the query horizon. A union of these sets across atoms is not a shared
    /// adjustment set.
    pub horizons: Option<CachedTemporalIdentification>,
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

enum DbnAtomOutcome {
    InvalidGraph,
    IdentifyFailed,
    NotIdentified,
    NoEstimand,
    Identified(CachedDbnPosteriorAtomIdentification),
}

/// Prepare-time identification for every atom in a DBN graph posterior.
#[derive(Clone, Debug)]
pub(crate) struct CachedDbnPosteriorIdentification {
    /// Frozen weights and keys; mediation flags describe eligibility at any
    /// cached horizon until projected with `mediation_horizon`.
    pub graphs: WeightedGraphSamples,
    /// Identified atoms, in posterior order. Unidentified atoms remain in
    /// [`Self::graphs`] with [`GraphIdentFlag::Unidentified`].
    pub atoms: Arc<[CachedDbnPosteriorAtomIdentification]>,
    /// Identification-time demotions for contrasts or a projected mediation
    /// horizon. Unprojected mediation uses `horizon_demotions` instead.
    pub identify_demotion: DbnIdentifyDemotion,
    /// Mediation failure counts for each requested horizon. Empty for contrasts.
    pub horizon_demotions: Arc<[(u32, DbnIdentifyDemotion)]>,
}

impl CachedDbnPosteriorIdentification {
    /// Project the cached union of eligible mediation atoms onto one horizon.
    /// An atom's failure at another horizon never changes this horizon's mass.
    pub fn mediation_horizon(&self, horizon: u32) -> Result<Self, CausalError> {
        let identify_demotion = self
            .horizon_demotions
            .iter()
            .find(|(h, _)| *h == horizon)
            .map(|(_, counts)| counts.clone())
            .ok_or_else(|| CausalError::Compile {
                message: format!("DBN mediation cache missing I({horizon})"),
            })?;
        let atoms: Vec<_> = self
            .atoms
            .iter()
            .filter_map(|atom| {
                let entry = atom.horizons.as_ref()?.get(horizon)?.clone();
                Some(CachedDbnPosteriorAtomIdentification {
                    key: atom.key,
                    estimand: entry.estimand.clone(),
                    identification: entry.identification.clone(),
                    indexer: entry.indexer.clone(),
                    horizons: Some(CachedTemporalIdentification { by_horizon: Arc::from([entry]) }),
                })
            })
            .collect();
        let eligible: std::collections::HashSet<_> = atoms.iter().map(|atom| atom.key).collect();
        let flags: Vec<_> = self
            .graphs
            .graph_keys
            .iter()
            .map(|key| {
                if eligible.contains(key) {
                    GraphIdentFlag::Identified
                } else {
                    GraphIdentFlag::Unidentified
                }
            })
            .collect();
        let graphs = WeightedGraphSamples::new(
            Arc::clone(&self.graphs.weights),
            flags,
            Arc::clone(&self.graphs.graph_keys),
        )
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
        let horizon_demotions = Arc::from([(horizon, identify_demotion.clone())]);
        Ok(Self { graphs, atoms: atoms.into(), identify_demotion, horizon_demotions })
    }
}

/// Class-envelope identification for one TemporalCpdag/Pag posterior atom.
#[derive(Clone, Debug)]
pub(crate) struct CachedTemporalClassPosteriorAtomIdentification {
    /// Collision-free, position-derived key used by the effect envelope.
    pub key: u64,
    /// Envelope-level identification (status, diagnostics, assumptions).
    pub identification: IdentificationResult,
    /// Shared estimand when every identified completion agrees.
    pub invariant: Option<IdentifiedEstimand>,
    /// Completions + unfold indexers for this class atom.
    pub envelope: antecedent_identify::TemporalClassEnvelope,
    /// Envelope identified weight before posterior mixing.
    pub identified_weight: f64,
    /// Completions whose search was capped.
    pub truncated_completions: usize,
}

/// Prepare-time identification for TemporalCpdag/Pag graph-posterior Pulse/Sustained.
#[derive(Clone, Debug)]
pub(crate) struct CachedTemporalClassPosteriorIdentification {
    /// Frozen weights, graph keys, and identified/unidentified flags.
    pub graphs: WeightedGraphSamples,
    /// Identified class atoms. Unidentified atoms remain in [`Self::graphs`].
    pub class_atoms: Arc<[CachedTemporalClassPosteriorAtomIdentification]>,
}

/// Identify unique adjacency masks under `ctx.parallelism`, then reduce in graph order.
fn identify_unique_adjacency_masks<T, F>(
    posterior: &GraphPosterior,
    ctx: &ExecutionContext,
    identify: F,
) -> Result<std::collections::HashMap<u64, T>, CausalError>
where
    T: Send,
    F: Fn(u64, &ExecutionContext) -> Result<T, CausalError> + Sync,
{
    let mut unique = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for i in 0..posterior.n_graphs {
        let mask = posterior.adjacency[i];
        if seen.insert(mask) {
            unique.push(mask);
        }
    }
    let values = ctx.map_indexed(unique.len(), |i, inner| identify(unique[i], inner))?;
    Ok(unique.into_iter().zip(values).collect())
}

/// One result per posterior position, under `ctx.parallelism`.
fn map_posterior_graphs<T, F>(
    posterior: &GraphPosterior,
    ctx: &ExecutionContext,
    f: F,
) -> Result<Vec<T>, CausalError>
where
    T: Send,
    F: Fn(usize, &ExecutionContext) -> Result<T, CausalError> + Sync,
{
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
    }
    ctx.map_indexed(posterior.n_graphs, |i, inner| {
        if inner.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
        }
        f(i, inner)
    })
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
    match posterior.atom_kind {
        antecedent_discovery::GraphPosteriorAtomKind::Dag => {}
        antecedent_discovery::GraphPosteriorAtomKind::Admg => {
            return build_admg_graph_posterior_identification_cache(posterior, query, ctx);
        }
        _ => return build_class_graph_posterior_identification_cache(posterior, query, ctx),
    }
    super::execute::report_identify_compute(ctx);
    let mut weights = Vec::with_capacity(posterior.n_graphs);
    let mut flags = Vec::with_capacity(posterior.n_graphs);
    let mut keys = Vec::with_capacity(posterior.n_graphs);
    let mut atoms = Vec::new();
    let by_mask = identify_unique_adjacency_masks(posterior, ctx, |mask, _inner| {
        (|| -> Result<Option<(IdentifiedEstimand, IdentificationResult)>, CausalError> {
            let Ok(dag) = dag_from_adjacency_mask(mask, posterior.n_vars) else {
                return Ok(None);
            };
            let Ok(identification) = identify_static(DEFAULT_IDENTIFIER_ID, &dag, query) else {
                return Ok(None);
            };
            if !super::execute::identification_status_ok_for_case(identification.status)
                || identification.estimands.is_empty()
            {
                return Ok(None);
            }
            let Ok(estimand) = select_estimand(&identification, EstimatorId::LinearAdjustmentAte)
                .or_else(|_| select_estimand(&identification, EstimatorId::BayesianGcomp))
            else {
                return Ok(None);
            };
            Ok(Some((estimand, identification)))
        })()
    })?;
    let mut atom_masks: HashMap<u64, u64> = HashMap::new();

    for i in 0..posterior.n_graphs {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
        }
        if let Some(progress) = &ctx.progress {
            #[allow(clippy::cast_precision_loss)]
            progress.report(i as f64 / posterior.n_graphs.max(1) as f64, "envelope.identify");
        }
        let mask = posterior.adjacency[i];
        let key = posterior.graph_keys[i];
        keys.push(key);
        weights.push(posterior.weights[i]);
        let resolved = by_mask.get(&mask).cloned().flatten();
        // A posterior may list the same graph more than once (one entry per
        // sample). Every entry keeps its own weight and flag in `graphs`, but
        // consumers weight an atom by the combined mass of its key, so each
        // key contributes exactly one atom.
        let first_for_key = match atom_masks.entry(key) {
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(mask);
                true
            }
            std::collections::hash_map::Entry::Occupied(slot) if *slot.get() == mask => false,
            std::collections::hash_map::Entry::Occupied(_) => {
                return Err(CausalError::Compile {
                    message: "graph posterior reuses one graph key for different adjacency masks"
                        .into(),
                });
            }
        };
        if let Some((estimand, identification)) = resolved {
            flags.push(GraphIdentFlag::Identified);
            if first_for_key {
                atoms.push(CachedGraphPosteriorAtomIdentification {
                    key,
                    estimand,
                    identification,
                });
            }
        } else {
            flags.push(GraphIdentFlag::Unidentified);
        }
    }
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
    }

    let graphs = WeightedGraphSamples::new(weights, flags, keys)
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
    Ok(CachedGraphPosteriorIdentification {
        graphs,
        atoms: Arc::from(atoms),
        class_atoms: Arc::from([]),
    })
}

/// Identify every ADMG posterior atom with general ID + functional.effect.
///
/// ADMG atoms are single graphs, not MEC completions. Unidentified mass stays
/// on the atom; there is no completion enumeration to mix.
/// Identify every ADMG posterior atom with general ID + functional.effect for a
/// static response query.
pub(crate) fn build_admg_graph_posterior_response_identification_cache(
    posterior: &GraphPosterior,
    query: &ResponseQuery,
    ctx: &ExecutionContext,
) -> Result<CachedGraphPosteriorIdentification, CausalError> {
    use std::collections::HashMap;

    use crate::strategy_table::{
        DEFAULT_ADMG_IDENTIFIER_ID, EstimatorId, identify_admg_query, select_estimand,
    };
    use antecedent_discovery::admg_from_adjacency_mask;

    super::execute::report_identify_compute(ctx);
    let mut weights = Vec::with_capacity(posterior.n_graphs);
    let mut flags = Vec::with_capacity(posterior.n_graphs);
    let mut keys = Vec::with_capacity(posterior.n_graphs);
    let mut atoms = Vec::new();
    // `estimate_admg_posterior_atom_response` reuses this atom's cached
    // identification/estimand verbatim as the *first grid level's* claim (it
    // only re-identifies per level from the second level on). A MeanCurve
    // query identified whole produces one general.id estimand per grid
    // level, and `select_estimand` then has no unique estimator match to
    // pick among them (they all report the same method). Cache the first
    // level's InterventionResponse claim instead, which is what downstream
    // code actually consumes and — being a single intervention level — is
    // exactly what `select_estimand` can disambiguate.
    let causal_query = match &query.functional {
        antecedent_core::ResponseFunctional::MeanCurve { outcome, treatment } => {
            let first_level = treatment
                .grid
                .values()
                .map_err(|e| CausalError::Compile { message: e.to_string() })?
                .into_iter()
                .next()
                .ok_or_else(|| CausalError::Compile {
                    message: "MeanCurve response requires a non-empty evaluation grid".into(),
                })?;
            let mut level_query = query.clone();
            level_query.functional = antecedent_core::ResponseFunctional::InterventionResponse {
                outcome: *outcome,
                interventions: Arc::from([Intervention::set(
                    treatment.variable,
                    Value::f64(first_level),
                )]),
            };
            CausalQuery::Response(level_query)
        }
        _ => CausalQuery::Response(query.clone()),
    };
    let by_mask = identify_unique_adjacency_masks(posterior, ctx, |mask, _inner| {
        (|| -> Result<Option<(IdentifiedEstimand, IdentificationResult)>, CausalError> {
            let Ok(admg) = admg_from_adjacency_mask(mask, posterior.n_vars) else {
                return Ok(None);
            };
            let Ok(identification) =
                identify_admg_query(DEFAULT_ADMG_IDENTIFIER_ID, &admg, &causal_query)
            else {
                return Ok(None);
            };
            if !super::execute::identification_status_ok_for_case(identification.status)
                || identification.estimands.is_empty()
            {
                return Ok(None);
            }
            let Ok(estimand) = select_estimand(&identification, EstimatorId::FunctionalEffect)
            else {
                return Ok(None);
            };
            Ok(Some((estimand, identification)))
        })()
    })?;
    let mut atom_masks: HashMap<u64, u64> = HashMap::new();

    for i in 0..posterior.n_graphs {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
        }
        if let Some(progress) = &ctx.progress {
            #[allow(clippy::cast_precision_loss)]
            progress.report(i as f64 / posterior.n_graphs.max(1) as f64, "envelope.identify");
        }
        let mask = posterior.adjacency[i];
        let key = posterior.graph_keys[i];
        keys.push(key);
        weights.push(posterior.weights[i]);
        let resolved = by_mask.get(&mask).cloned().flatten();
        let first_for_key = match atom_masks.entry(key) {
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(mask);
                true
            }
            std::collections::hash_map::Entry::Occupied(slot) if *slot.get() == mask => false,
            std::collections::hash_map::Entry::Occupied(_) => {
                return Err(CausalError::Compile {
                    message: "graph posterior reuses one graph key for different adjacency masks"
                        .into(),
                });
            }
        };
        if let Some((estimand, identification)) = resolved {
            flags.push(GraphIdentFlag::Identified);
            if first_for_key {
                atoms.push(CachedGraphPosteriorAtomIdentification {
                    key,
                    estimand,
                    identification,
                });
            }
        } else {
            flags.push(GraphIdentFlag::Unidentified);
        }
    }
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
    }

    let graphs = WeightedGraphSamples::new(weights, flags, keys)
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
    Ok(CachedGraphPosteriorIdentification {
        graphs,
        atoms: Arc::from(atoms),
        class_atoms: Arc::from([]),
    })
}

fn build_admg_graph_posterior_identification_cache(
    posterior: &GraphPosterior,
    query: &AverageEffectQuery,
    ctx: &ExecutionContext,
) -> Result<CachedGraphPosteriorIdentification, CausalError> {
    use std::collections::HashMap;

    use crate::strategy_table::{
        DEFAULT_ADMG_IDENTIFIER_ID, EstimatorId, identify_admg, select_estimand,
    };
    use antecedent_discovery::admg_from_adjacency_mask;

    super::execute::report_identify_compute(ctx);
    let mut weights = Vec::with_capacity(posterior.n_graphs);
    let mut flags = Vec::with_capacity(posterior.n_graphs);
    let mut keys = Vec::with_capacity(posterior.n_graphs);
    let mut atoms = Vec::new();
    let by_mask = identify_unique_adjacency_masks(posterior, ctx, |mask, _inner| {
        (|| -> Result<Option<(IdentifiedEstimand, IdentificationResult)>, CausalError> {
            let Ok(admg) = admg_from_adjacency_mask(mask, posterior.n_vars) else {
                return Ok(None);
            };
            let Ok(identification) = identify_admg(DEFAULT_ADMG_IDENTIFIER_ID, &admg, query) else {
                return Ok(None);
            };
            if !super::execute::identification_status_ok_for_case(identification.status)
                || identification.estimands.is_empty()
            {
                return Ok(None);
            }
            let Ok(estimand) = select_estimand(&identification, EstimatorId::FunctionalEffect)
            else {
                return Ok(None);
            };
            Ok(Some((estimand, identification)))
        })()
    })?;
    let mut atom_masks: HashMap<u64, u64> = HashMap::new();

    for i in 0..posterior.n_graphs {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
        }
        if let Some(progress) = &ctx.progress {
            #[allow(clippy::cast_precision_loss)]
            progress.report(i as f64 / posterior.n_graphs.max(1) as f64, "envelope.identify");
        }
        let mask = posterior.adjacency[i];
        let key = posterior.graph_keys[i];
        keys.push(key);
        weights.push(posterior.weights[i]);
        let resolved = by_mask.get(&mask).cloned().flatten();
        let first_for_key = match atom_masks.entry(key) {
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(mask);
                true
            }
            std::collections::hash_map::Entry::Occupied(slot) if *slot.get() == mask => false,
            std::collections::hash_map::Entry::Occupied(_) => {
                return Err(CausalError::Compile {
                    message: "graph posterior reuses one graph key for different adjacency masks"
                        .into(),
                });
            }
        };
        if let Some((estimand, identification)) = resolved {
            flags.push(GraphIdentFlag::Identified);
            if first_for_key {
                atoms.push(CachedGraphPosteriorAtomIdentification {
                    key,
                    estimand,
                    identification,
                });
            }
        } else {
            flags.push(GraphIdentFlag::Unidentified);
        }
    }
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
    }

    let graphs = WeightedGraphSamples::new(weights, flags, keys)
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
    Ok(CachedGraphPosteriorIdentification {
        graphs,
        atoms: Arc::from(atoms),
        class_atoms: Arc::from([]),
    })
}

/// Identify every CPDAG/PAG posterior atom with the existing class ATE envelope.
fn build_class_graph_posterior_identification_cache(
    posterior: &GraphPosterior,
    query: &AverageEffectQuery,
    ctx: &ExecutionContext,
) -> Result<CachedGraphPosteriorIdentification, CausalError> {
    use std::collections::HashMap;

    use crate::strategy_table::{DEFAULT_PAG_IDENTIFIER_ID, identify_cpdag, identify_pag};
    use antecedent_discovery::{cpdag_from_adjacency_mask, pag_from_adjacency_mask};

    super::execute::report_identify_compute(ctx);
    let resolved = map_posterior_graphs(posterior, ctx, |i, _inner| {
        let mask = posterior.adjacency[i];
        let mark = posterior.mark_masks.as_ref().map_or(0, |marks| marks[i]);
        let key = posterior.graph_keys[i];
        let cached = match posterior.atom_kind {
            antecedent_discovery::GraphPosteriorAtomKind::Cpdag => {
                let Ok(cpdag) = cpdag_from_adjacency_mask(mask, posterior.n_vars) else {
                    return Ok(None);
                };
                Ok::<_, CausalError>(
                    identify_cpdag(DEFAULT_PAG_IDENTIFIER_ID, &cpdag, query)
                        .ok()
                        .map(|envelope| cache_class_envelope(key, query, envelope)),
                )
            }
            antecedent_discovery::GraphPosteriorAtomKind::Pag => {
                let Ok(pag) = pag_from_adjacency_mask(mask, mark, posterior.n_vars) else {
                    return Ok(None);
                };
                Ok(identify_pag(DEFAULT_PAG_IDENTIFIER_ID, &pag, query)
                    .ok()
                    .map(|envelope| cache_class_envelope(key, query, envelope)))
            }
            _ => Ok(None),
        }?;
        Ok(cached)
    })?;
    let mut weights = Vec::with_capacity(posterior.n_graphs);
    let mut flags = Vec::with_capacity(posterior.n_graphs);
    let mut keys = Vec::with_capacity(posterior.n_graphs);
    let mut class_atoms = Vec::new();
    let mut atom_masks: HashMap<u64, u64> = HashMap::new();

    for (i, cached) in resolved.into_iter().enumerate() {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
        }
        if let Some(progress) = &ctx.progress {
            #[allow(clippy::cast_precision_loss)]
            progress.report(i as f64 / posterior.n_graphs.max(1) as f64, "envelope.identify");
        }
        let mask = posterior.adjacency[i];
        let key = posterior.graph_keys[i];
        keys.push(key);
        weights.push(posterior.weights[i]);
        let first_for_key = match atom_masks.entry(key) {
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(mask);
                true
            }
            std::collections::hash_map::Entry::Occupied(slot) if *slot.get() == mask => false,
            std::collections::hash_map::Entry::Occupied(_) => {
                return Err(CausalError::Compile {
                    message: "graph posterior reuses one graph key for different adjacency masks"
                        .into(),
                });
            }
        };
        let Some(cached) = cached else {
            flags.push(GraphIdentFlag::Unidentified);
            continue;
        };
        // Envelope-level GraphDependent still carries identified completion mass.
        // Only atoms with zero identified completion weight are posterior-unidentified.
        let identified = cached.identified_weight > 0.0;
        if identified {
            flags.push(GraphIdentFlag::Identified);
        } else {
            flags.push(GraphIdentFlag::Unidentified);
        }
        if first_for_key && identified {
            class_atoms.push(cached);
        }
    }
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
    }
    let graphs = WeightedGraphSamples::new(weights, flags, keys)
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
    Ok(CachedGraphPosteriorIdentification {
        graphs,
        atoms: Arc::from([]),
        class_atoms: Arc::from(class_atoms),
    })
}

fn cache_class_envelope<G>(
    key: u64,
    query: &AverageEffectQuery,
    envelope: IdentificationEnvelope<G>,
) -> CachedClassPosteriorAtomIdentification {
    use crate::strategy_table::{EstimatorId, select_estimand};

    let identification = super::execute::envelope_to_identification_result(&envelope, query);
    let cases: Vec<CachedClassPosteriorCase> = envelope
        .cases
        .iter()
        .map(|case| {
            let estimand = if super::execute::identification_status_ok_for_case(case.result.status)
                && !case.result.estimands.is_empty()
            {
                select_estimand(&case.result, EstimatorId::LinearAdjustmentAte)
                    .or_else(|_| select_estimand(&case.result, EstimatorId::BayesianGcomp))
                    .ok()
                    .or_else(|| case.result.estimands.first().cloned())
            } else {
                None
            };
            CachedClassPosteriorCase { weight: case.weight.0, status: case.result.status, estimand }
        })
        .collect();
    CachedClassPosteriorAtomIdentification {
        key,
        identification,
        invariant: envelope.invariant,
        cases: Arc::from(cases),
        identified_weight: envelope.identified_weight.0,
        truncated_completions: envelope.truncated_completions,
    }
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
    use crate::strategy_table::select_estimand;

    let lag_masks = posterior.lag_masks.as_ref().ok_or_else(|| CausalError::Compile {
        message: "DBN posterior missing per-atom lag masks".into(),
    })?;
    let max_lag = posterior
        .max_lag
        .ok_or_else(|| CausalError::Compile { message: "DBN posterior missing max_lag".into() })?;
    super::execute::report_identify_compute(ctx);
    let mapped = map_posterior_graphs(posterior, ctx, |i, _inner| {
        // A DBN atom is the pair (contemporaneous mask, lag mask), but the
        // public GraphPosterior constructor keys atoms by contemporaneous mask
        // alone. Use posterior position as a collision-free execution key so
        // lag-distinct atoms keep distinct fits, flags, and weights inside the
        // effect envelope. This key is internal and does not alter the public
        // GraphPosterior representation.
        let key = dbn_envelope_key(i)?;
        let Ok(graph) = temporal_dag_from_dbn_masks(
            posterior.adjacency[i],
            lag_masks[i],
            posterior.n_vars,
            max_lag,
            variables,
        ) else {
            return Ok(DbnAtomOutcome::InvalidGraph);
        };
        let Ok(temporal) = TemporalBackdoorIdentifier::new().identify_temporal(&graph, query)
        else {
            return Ok(DbnAtomOutcome::IdentifyFailed);
        };
        let identification = temporal.result;
        if !super::execute::identification_status_ok_for_case(identification.status) {
            return Ok(DbnAtomOutcome::NotIdentified);
        }
        if identification.estimands.is_empty() {
            return Ok(DbnAtomOutcome::NoEstimand);
        }
        let estimator = dbn_temporal_effect_estimator(query);
        let Ok(estimand) = select_estimand(&identification, estimator) else {
            return Ok(DbnAtomOutcome::NoEstimand);
        };
        Ok(DbnAtomOutcome::Identified(CachedDbnPosteriorAtomIdentification {
            key,
            estimand,
            identification,
            indexer: temporal.indexer,
            horizons: None,
        }))
    })?;
    let mut weights = Vec::with_capacity(posterior.n_graphs);
    let mut flags = Vec::with_capacity(posterior.n_graphs);
    let mut keys = Vec::with_capacity(posterior.n_graphs);
    let mut atoms = Vec::new();
    let mut identify_demotion = DbnIdentifyDemotion::default();

    for (i, outcome) in mapped.into_iter().enumerate() {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
        }
        if let Some(progress) = &ctx.progress {
            #[allow(clippy::cast_precision_loss)]
            progress.report(i as f64 / posterior.n_graphs.max(1) as f64, "envelope.identify");
        }
        let key = dbn_envelope_key(i)?;
        keys.push(key);
        weights.push(posterior.weights[i]);
        match outcome {
            DbnAtomOutcome::InvalidGraph => {
                flags.push(GraphIdentFlag::Unidentified);
                identify_demotion.invalid_graph += 1;
            }
            DbnAtomOutcome::IdentifyFailed => {
                flags.push(GraphIdentFlag::Unidentified);
                identify_demotion.identify_failed += 1;
            }
            DbnAtomOutcome::NotIdentified => {
                flags.push(GraphIdentFlag::Unidentified);
                identify_demotion.not_identified += 1;
            }
            DbnAtomOutcome::NoEstimand => {
                flags.push(GraphIdentFlag::Unidentified);
                identify_demotion.no_estimand += 1;
            }
            DbnAtomOutcome::Identified(atom) => {
                flags.push(GraphIdentFlag::Identified);
                atoms.push(atom);
            }
        }
    }
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
    }

    let graphs = WeightedGraphSamples::new(weights, flags, keys)
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
    Ok(CachedDbnPosteriorIdentification {
        graphs,
        atoms: Arc::from(atoms),
        identify_demotion,
        horizon_demotions: Arc::from([]),
    })
}

/// Identify every TemporalCpdag/Pag posterior atom with the class envelope.
///
/// Each atom is reconstructed from adjacency + lag masks (+ mark masks for
/// Pag). Completions that disagree keep envelope status; unidentified class
/// mass stays on [`CachedTemporalClassPosteriorIdentification::graphs`].
/// Completion enumeration is not posterior probability.
pub(crate) fn build_temporal_class_posterior_identification_cache(
    posterior: &GraphPosterior,
    variables: &[antecedent_core::VariableId],
    query: &TemporalEffectQuery,
    max_completions: Option<usize>,
    ctx: &ExecutionContext,
) -> Result<CachedTemporalClassPosteriorIdentification, CausalError> {
    use crate::strategy_table::{
        DEFAULT_PAG_IDENTIFIER_ID, identify_temporal_cpdag_configured,
        identify_temporal_pag_configured,
    };

    let lag_masks = posterior.lag_masks.as_ref().ok_or_else(|| CausalError::Compile {
        message: "temporal class posterior missing per-atom lag masks".into(),
    })?;
    let max_lag = posterior.max_lag.ok_or_else(|| CausalError::Compile {
        message: "temporal class posterior missing max_lag".into(),
    })?;
    if !matches!(
        posterior.atom_kind,
        antecedent_discovery::GraphPosteriorAtomKind::Cpdag
            | antecedent_discovery::GraphPosteriorAtomKind::Pag
    ) {
        return Err(CausalError::Compile {
            message: "temporal class posterior requires Cpdag or Pag atom_kind".into(),
        });
    }
    super::execute::report_identify_compute(ctx);
    let mut config = antecedent_identify::GeneralizedAdjustmentConfig::default();
    if let Some(max) = max_completions {
        config.max_completions = max;
    }
    let mapped = map_posterior_graphs(posterior, ctx, |i, _inner| {
        let key = dbn_envelope_key(i)?;
        let mark = posterior.mark_masks.as_ref().map_or(0, |marks| marks[i]);
        let cached = match posterior.atom_kind {
            antecedent_discovery::GraphPosteriorAtomKind::Cpdag => {
                let Ok(cpdag) = temporal_cpdag_from_dbn_masks(
                    posterior.adjacency[i],
                    lag_masks[i],
                    posterior.n_vars,
                    max_lag,
                    variables,
                ) else {
                    return Ok(None);
                };
                Ok::<_, CausalError>(
                    identify_temporal_cpdag_configured(
                        DEFAULT_PAG_IDENTIFIER_ID,
                        &cpdag,
                        query,
                        config.clone(),
                    )
                    .ok()
                    .map(|envelope| cache_temporal_class_atom(key, query, envelope)),
                )
            }
            antecedent_discovery::GraphPosteriorAtomKind::Pag => {
                let Ok(pag) = temporal_pag_from_dbn_masks(
                    posterior.adjacency[i],
                    lag_masks[i],
                    mark,
                    posterior.n_vars,
                    max_lag,
                    variables,
                ) else {
                    return Ok(None);
                };
                Ok(identify_temporal_pag_configured(
                    DEFAULT_PAG_IDENTIFIER_ID,
                    &pag,
                    query,
                    config.clone(),
                )
                .ok()
                .map(|envelope| cache_temporal_class_atom(key, query, envelope)))
            }
            _ => Ok(None),
        }?;
        Ok(cached)
    })?;
    let mut weights = Vec::with_capacity(posterior.n_graphs);
    let mut flags = Vec::with_capacity(posterior.n_graphs);
    let mut keys = Vec::with_capacity(posterior.n_graphs);
    let mut class_atoms = Vec::new();

    for (i, cached) in mapped.into_iter().enumerate() {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
        }
        if let Some(progress) = &ctx.progress {
            #[allow(clippy::cast_precision_loss)]
            progress.report(i as f64 / posterior.n_graphs.max(1) as f64, "envelope.identify");
        }
        let key = dbn_envelope_key(i)?;
        keys.push(key);
        weights.push(posterior.weights[i]);
        let Some(cached) = cached else {
            flags.push(GraphIdentFlag::Unidentified);
            continue;
        };
        if cached.identified_weight > 0.0 {
            flags.push(GraphIdentFlag::Identified);
            class_atoms.push(cached);
        } else {
            flags.push(GraphIdentFlag::Unidentified);
        }
    }
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
    }
    let graphs = WeightedGraphSamples::new(weights, flags, keys)
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
    Ok(CachedTemporalClassPosteriorIdentification { graphs, class_atoms: Arc::from(class_atoms) })
}

fn cache_temporal_class_atom(
    key: u64,
    query: &TemporalEffectQuery,
    envelope: antecedent_identify::TemporalClassEnvelope,
) -> CachedTemporalClassPosteriorAtomIdentification {
    let identification = super::execute::envelope_to_identification_result_for(
        &envelope.envelope,
        CausalQuery::TemporalEffect(query.clone()),
    );
    CachedTemporalClassPosteriorAtomIdentification {
        key,
        identification,
        invariant: envelope.envelope.invariant.clone(),
        identified_weight: envelope.envelope.identified_weight.0,
        truncated_completions: envelope.envelope.truncated_completions,
        envelope,
    }
}

/// Identify every DBN atom for a temporal [`CausalQuery::Response`].
///
/// Each atom is reconstructed as a [`TemporalDag`] and identified at every
/// requested horizon. An atom that fails any horizon stays unidentified;
/// unidentified mass is retained. Atoms are DAGs: there is no completion
/// enumeration.
pub(crate) fn build_dbn_posterior_response_identification_cache(
    posterior: &GraphPosterior,
    variables: &[antecedent_core::VariableId],
    query: &ResponseQuery,
    estimator_id: crate::strategy_table::EstimatorId,
    ctx: &ExecutionContext,
) -> Result<CachedDbnPosteriorIdentification, CausalError> {
    super::execute::dbn_posterior_response_supported(query)?;
    let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
        message: "DBN-posterior response requires TemporalResponseSpec".into(),
    })?;
    let (treatment, outcome) = query.functional.primary_pair().ok_or_else(|| {
        CausalError::Compile { message: "response query has no treatment/outcome pair".into() }
    })?;
    let lag_masks = posterior.lag_masks.as_ref().ok_or_else(|| CausalError::Compile {
        message: "DBN posterior missing per-atom lag masks".into(),
    })?;
    let max_lag = posterior
        .max_lag
        .ok_or_else(|| CausalError::Compile { message: "DBN posterior missing max_lag".into() })?;
    super::execute::report_identify_compute(ctx);
    let mapped = map_posterior_graphs(posterior, ctx, |i, _inner| {
        let key = dbn_envelope_key(i)?;
        let Ok(graph) = temporal_dag_from_dbn_masks(
            posterior.adjacency[i],
            lag_masks[i],
            posterior.n_vars,
            max_lag,
            variables,
        ) else {
            return Ok(DbnAtomOutcome::InvalidGraph);
        };
        let Ok(horizons) = identify_temporal_response_horizons(
            &graph,
            treatment,
            outcome,
            temporal,
            &query.target_population,
            estimator_id,
            None,
            single_step_dose(query).ok().flatten(),
        ) else {
            return Ok(DbnAtomOutcome::IdentifyFailed);
        };
        if horizons.by_horizon.len() != temporal.horizons.len()
            || horizons.by_horizon.iter().any(|entry| {
                !super::execute::identification_status_ok_for_case(entry.identification.status)
                    || entry.identification.estimands.is_empty()
            })
        {
            return Ok(DbnAtomOutcome::NotIdentified);
        }
        let Some(first) = horizons.by_horizon.first() else {
            return Ok(DbnAtomOutcome::NoEstimand);
        };
        Ok(DbnAtomOutcome::Identified(CachedDbnPosteriorAtomIdentification {
            key,
            estimand: first.estimand.clone(),
            identification: first.identification.clone(),
            indexer: first.indexer.clone(),
            horizons: Some(horizons),
        }))
    })?;
    let mut weights = Vec::with_capacity(posterior.n_graphs);
    let mut flags = Vec::with_capacity(posterior.n_graphs);
    let mut keys = Vec::with_capacity(posterior.n_graphs);
    let mut atoms = Vec::new();
    let mut identify_demotion = DbnIdentifyDemotion::default();

    for (i, outcome) in mapped.into_iter().enumerate() {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
        }
        if let Some(progress) = &ctx.progress {
            #[allow(clippy::cast_precision_loss)]
            progress.report(i as f64 / posterior.n_graphs.max(1) as f64, "envelope.identify");
        }
        let key = dbn_envelope_key(i)?;
        keys.push(key);
        weights.push(posterior.weights[i]);
        match outcome {
            DbnAtomOutcome::InvalidGraph => {
                flags.push(GraphIdentFlag::Unidentified);
                identify_demotion.invalid_graph += 1;
            }
            DbnAtomOutcome::IdentifyFailed => {
                flags.push(GraphIdentFlag::Unidentified);
                identify_demotion.identify_failed += 1;
            }
            DbnAtomOutcome::NotIdentified => {
                flags.push(GraphIdentFlag::Unidentified);
                identify_demotion.not_identified += 1;
            }
            DbnAtomOutcome::NoEstimand => {
                flags.push(GraphIdentFlag::Unidentified);
                identify_demotion.no_estimand += 1;
            }
            DbnAtomOutcome::Identified(atom) => {
                flags.push(GraphIdentFlag::Identified);
                atoms.push(atom);
            }
        }
    }
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
    }

    let graphs = WeightedGraphSamples::new(weights, flags, keys)
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
    Ok(CachedDbnPosteriorIdentification {
        graphs,
        atoms: Arc::from(atoms),
        identify_demotion,
        horizon_demotions: Arc::from([]),
    })
}

/// Identify every DBN atom for [`CausalQuery::Mediation`] (`TemporalMediationEffect`).
///
/// Each atom gets its own `I(h)` cache from that atom's reconstructed
/// [`TemporalDag`]. Adjustment sets are not unioned across atoms or horizons.
/// Identification failures stay unidentified; priors are not consulted.
pub(crate) fn build_dbn_posterior_mediation_identification_cache(
    posterior: &GraphPosterior,
    variables: &[antecedent_core::VariableId],
    query: &MediationQuery,
    ctx: &ExecutionContext,
) -> Result<CachedDbnPosteriorIdentification, CausalError> {
    build_dbn_mediation_cache_with_identifier(
        posterior,
        variables,
        query,
        ctx,
        |_, _, graph, horizon_query| {
            identify_temporal_mediation_horizons(
                graph,
                horizon_query,
                crate::strategy_table::EstimatorId::BayesianTemporalMediation,
            )
        },
    )
}

fn build_dbn_mediation_cache_with_identifier(
    posterior: &GraphPosterior,
    variables: &[antecedent_core::VariableId],
    query: &MediationQuery,
    ctx: &ExecutionContext,
    mut identify: impl FnMut(
        usize,
        u32,
        &TemporalDag,
        &MediationQuery,
    ) -> Result<CachedTemporalIdentification, CausalError>,
) -> Result<CachedDbnPosteriorIdentification, CausalError> {
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
    query.validate().map_err(|error| CausalError::Compile { message: error.to_string() })?;
    let mut horizon_demotions: Vec<_> =
        query.horizons.iter().map(|h| (*h, DbnIdentifyDemotion::default())).collect();

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
            for (_, counts) in &mut horizon_demotions {
                counts.invalid_graph += 1;
            }
            continue;
        };
        let mut entries = Vec::new();
        for (horizon, counts) in &mut horizon_demotions {
            let mut horizon_query = query.clone();
            horizon_query.horizons = Arc::from([*horizon]);
            let Ok(horizons) = identify(i, *horizon, &graph, &horizon_query) else {
                counts.identify_failed += 1;
                continue;
            };
            let Some(entry) = horizons.by_horizon.first() else {
                counts.no_estimand += 1;
                continue;
            };
            if !super::execute::identification_status_ok_for_case(entry.identification.status)
                || entry.identification.estimands.is_empty()
            {
                counts.not_identified += 1;
                continue;
            }
            entries.push(entry.clone());
        }
        let Some(first) = entries.first() else {
            flags.push(GraphIdentFlag::Unidentified);
            continue;
        };
        flags.push(GraphIdentFlag::Identified);
        atoms.push(CachedDbnPosteriorAtomIdentification {
            key,
            estimand: first.estimand.clone(),
            identification: first.identification.clone(),
            indexer: first.indexer.clone(),
            horizons: Some(CachedTemporalIdentification { by_horizon: entries.into() }),
        });
    }
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: super::stage::STAGE_IDENTIFY });
    }

    let graphs = WeightedGraphSamples::new(weights, flags, keys)
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
    Ok(CachedDbnPosteriorIdentification {
        graphs,
        atoms: atoms.into(),
        identify_demotion: DbnIdentifyDemotion::default(),
        horizon_demotions: horizon_demotions.into(),
    })
}

fn dbn_temporal_effect_estimator(
    query: &TemporalEffectQuery,
) -> crate::strategy_table::EstimatorId {
    use crate::strategy_table::EstimatorId;
    if query.is_multi_step_sustained() {
        EstimatorId::TemporalSequentialGcomp
    } else {
        EstimatorId::TemporalLinearAdjustment
    }
}

/// Reconstruct the [`TemporalDag`] for a DBN envelope key (posterior index).
pub(crate) fn temporal_dag_from_dbn_atom(
    posterior: &GraphPosterior,
    key: u64,
    variables: &[antecedent_core::VariableId],
) -> Result<TemporalDag, CausalError> {
    let index = usize::try_from(key).map_err(|_| CausalError::Compile {
        message: "DBN envelope key does not fit a posterior index".into(),
    })?;
    let lag_masks = posterior.lag_masks.as_ref().ok_or_else(|| CausalError::Compile {
        message: "DBN posterior missing per-atom lag masks".into(),
    })?;
    let max_lag = posterior
        .max_lag
        .ok_or_else(|| CausalError::Compile { message: "DBN posterior missing max_lag".into() })?;
    if index >= posterior.n_graphs || index >= lag_masks.len() {
        return Err(CausalError::Compile { message: "DBN envelope key is out of range".into() });
    }
    temporal_dag_from_dbn_masks(
        posterior.adjacency[index],
        lag_masks[index],
        posterior.n_vars,
        max_lag,
        variables,
    )
    .map_err(|error| CausalError::Compile { message: error.to_string() })
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
/// frozen for that query's horizon. For [`CausalQuery::Mediation`]
/// (`TemporalMediationEffect`), identification is frozen independently for
/// every requested horizon.
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

/// Data modality a [`PreparedStudy`] was compiled for.
///
/// The physical plan is modality-specific, so a handle prepared on one
/// modality must refuse data of another instead of running a different route
/// under the frozen plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PreparedModality {
    /// Tabular rows: [`PreparedStudy::estimate`] / [`PreparedStudy::refresh`].
    Tabular,
    /// One time series (or event data on a regular grid):
    /// [`PreparedStudy::estimate_series`] / [`PreparedStudy::refresh_series`].
    Series,
    /// Multi-unit panel: [`PreparedStudy::estimate_panel`] / [`PreparedStudy::refresh_panel`].
    Panel,
    /// Multi-environment series: [`PreparedStudy::estimate_multi_env`] /
    /// [`PreparedStudy::refresh_multi_env`].
    MultiEnv,
}

impl PreparedModality {
    const fn label(self) -> &'static str {
        match self {
            Self::Tabular => "tabular",
            Self::Series => "series",
            Self::Panel => "panel",
            Self::MultiEnv => "multi_env",
        }
    }

    const fn estimate_entry(self) -> &'static str {
        match self {
            Self::Tabular => "estimate",
            Self::Series => "estimate_series",
            Self::Panel => "estimate_panel",
            Self::MultiEnv => "estimate_multi_env",
        }
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
pub struct PreparedStudy<S = SampledPreparedState> {
    pub(crate) state: S,
}

/// Retained state for sampled-data modalities of the common prepared handle.
#[derive(Clone, Debug)]
pub struct SampledPreparedState {
    /// Frozen analysis config (data slot replaced on each estimate). Read
    /// through `PreparedStudy::study`; mutate only through `PreparedStudy::study_mut`, which
    /// drops the compiled program identities.
    analysis: Study,
    /// Data-independent contract layers, compiled once per handle state.
    program_cache: std::sync::OnceLock<Arc<super::contract::ProgramPayloads>>,
    /// Ready physical plan from the prepare-time compile (never recompiled on refresh).
    plan: PhysicalExecutionPlan,
    /// Schema fingerprint from prepare-time data.
    schema: CausalSchema,
    /// Data modality frozen at prepare; each estimate / refresh entry point
    /// accepts only its own modality.
    modality: PreparedModality,
    /// Sampling regularity frozen for series and panel prepares (`None` = tabular).
    time_regularity: Option<antecedent_data::SamplingRegularity>,
    /// Cross-fitted AIPW scores frozen at prepare when the cell can export them.
    score_table: Option<antecedent_estimate::ScoreTable>,
}

impl std::ops::Deref for PreparedStudy {
    type Target = SampledPreparedState;
    fn deref(&self) -> &Self::Target {
        &self.state
    }
}
impl std::ops::DerefMut for PreparedStudy {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state
    }
}

impl PreparedStudy {
    /// Borrow the caller's frozen query, with original variable ids and query kind.
    #[must_use]
    pub fn query(&self) -> &CausalQuery {
        &self.analysis.query
    }

    /// Crate-visible frozen study.
    pub(crate) fn study(&self) -> &Study {
        &self.analysis
    }

    /// Mutable frozen study. Drops the compiled program identities so the
    /// next contract or estimate recompiles them from the changed study.
    pub(crate) fn study_mut(&mut self) -> &mut Study {
        self.program_cache = std::sync::OnceLock::new();
        &mut self.analysis
    }

    /// Replace the frozen study after a successful refresh.
    fn replace_study(&mut self, study: Study) {
        *self.study_mut() = study;
    }

    pub(crate) fn program_cache(
        &self,
    ) -> &std::sync::OnceLock<Arc<super::contract::ProgramPayloads>> {
        &self.program_cache
    }

    /// Series data in the variant this handle was prepared with (event
    /// studies keep the event modality through estimate and refresh).
    fn series_input(&self, data: TimeSeriesData) -> DataInput {
        match self.analysis.data {
            DataInput::Event(_) => DataInput::Event(data),
            _ => DataInput::Temporal(data),
        }
    }

    /// Stamp the contract `result` was executed under on `data`.
    fn stamp(&self, data: &DataInput, mut result: StudyResult) -> Result<StudyResult, CausalError> {
        super::execute::push_gaussian_likelihood_disclosure(
            &mut result,
            &self.analysis.inference,
            data,
        );
        result.executed_contract =
            Some(self.executed_contract(data, self.analysis.refute, None)?);
        Ok(result)
    }

    /// Replace custom validators. Incoming names must match the prepare-time set.
    ///
    /// # Errors
    ///
    /// Name mismatch.
    pub fn rebind_custom_validators(
        &mut self,
        validators: Vec<std::sync::Arc<dyn antecedent_validate::CustomEffectValidator>>,
    ) -> Result<(), CausalError> {
        let expected: std::collections::BTreeSet<&str> =
            self.analysis.custom_validators.iter().map(|v| v.name()).collect();
        let incoming: std::collections::BTreeSet<&str> =
            validators.iter().map(|v| v.name()).collect();
        if expected != incoming {
            return Err(crate::unsupported_reason!(
                "attested_not_reverifiable",
                "rebind_validators names must match attested names"
            ));
        }
        self.analysis.custom_validators = validators;
        Ok(())
    }

    /// Population bindings frozen at prepare, when the query names a predicate
    /// or a custom target distribution.
    #[must_use]
    pub fn population_registry(&self) -> Option<&antecedent_core::PopulationRegistry> {
        self.analysis.population_registry.as_ref()
    }

    /// Names of the caller custom validators frozen at prepare, in order.
    ///
    /// These are the names a claim attests and the names
    /// [`Self::rebind_custom_validators`] requires.
    #[must_use]
    pub fn custom_validator_names(&self) -> Vec<&str> {
        self.analysis.custom_validators.iter().map(|validator| validator.name()).collect()
    }

    /// Stream progressive stages from later estimates of this handle.
    ///
    /// `None` stops streaming. The sink is an in-process execution control: it
    /// is not part of the contract, program, or claim identity.
    pub fn set_stage_sink(&mut self, sink: Option<Arc<dyn super::stage::StageResultSink>>) {
        self.analysis.stage_sink = sink;
    }

    /// Whether estimates of this handle stream identify → estimate_point →
    /// uncertainty → validate stage payloads.
    ///
    /// Stages describe one identification and one scalar effect estimate. They
    /// stream from the single-estimand static executors: a supplied or DAG-coerced
    /// DAG average effect (Frequentist or Bayesian, excluding `rd.sharp` and the
    /// general-ID functional plug-in) and the Bayesian DAG conditional effect.
    #[must_use]
    pub fn streams_stages(&self) -> bool {
        let analysis = &self.analysis;
        if analysis.graph_posterior.is_some()
            || analysis.tiered.is_some()
            || !matches!(analysis.data, DataInput::Tabular(_))
        {
            return false;
        }
        let dag_like = match analysis.graph.class() {
            GraphClass::Dag => true,
            GraphClass::Admg => analysis
                .graph
                .as_admg()
                .is_some_and(|admg| !super::execute::admg_has_bidirected(admg)),
            _ => false,
        };
        let estimator = self.plan.logical.record.estimator.as_deref();
        match &analysis.query {
            CausalQuery::AverageEffect(_) => {
                dag_like && !matches!(estimator, Some("rd.sharp" | "functional.effect"))
            }
            CausalQuery::ConditionalEffect(_) => {
                analysis.graph.class() == GraphClass::Dag
                    && matches!(analysis.inference, InferenceMode::Bayesian(_))
            }
            _ => false,
        }
    }

    /// Frozen horizon-specific identification and its exact unfolded variable namespace.
    ///
    /// A `TemporalDag` prepare caches this directly. A DBN posterior or a TemporalCpdag/Pag
    /// envelope caches a per-atom or per-completion result instead; this projects either
    /// onto the same shape (see [`super::contract::full_temporal_identification`]), so an
    /// exported `analysis_result` artifact validates its identification against the same
    /// namespace the compiled contract already uses.
    #[must_use]
    pub fn temporal_identification(
        &self,
    ) -> Option<std::borrow::Cow<'_, CachedTemporalIdentification>> {
        super::contract::full_temporal_identification(&self.analysis)
    }

    /// Borrow the frozen schema fingerprint.
    #[must_use]
    pub fn schema(&self) -> &CausalSchema {
        &self.schema
    }

    /// Matrix structure-source axis frozen at prepare.
    #[must_use]
    pub const fn structure_source(&self) -> crate::support::StructureSource {
        self.state.analysis.structure_source()
    }

    /// Evidence contract frozen at prepare. `None` when the query is off-axis.
    #[must_use]
    pub const fn support_status(&self) -> Option<crate::support::CellStatus> {
        self.state.analysis.support_status()
    }

    /// Borrow the ready physical plan retained from prepare.
    #[must_use]
    pub fn plan(&self) -> &PhysicalExecutionPlan {
        &self.plan
    }

    /// Borrow the prepare-time AIPW score table, when the cell exported one.
    #[must_use]
    pub fn score_table(&self) -> Option<&ScoreTable> {
        self.score_table.as_ref()
    }

    /// Shared batch design attached when this plan was prepared inside a batch.
    #[must_use]
    pub fn shared_design(&self) -> Option<&super::batch::SharedBatchDesign> {
        self.analysis.shared_batch_design.as_deref()
    }

    /// The prepared score table a retarget reweights.
    ///
    /// # Errors
    ///
    /// `score_table_unavailable` when prepare built none.
    fn retarget_score_table(&self) -> Result<&antecedent_estimate::ScoreTable, CausalError> {
        self.score_table.as_ref().ok_or_else(|| {
            crate::unsupported_reason!(
                "score_table_unavailable",
                "retarget requires a prepared score table on AverageEffect or discrete joint \
                 InterventionResponse"
            )
        })
    }

    /// Whether this handle can perform `intent` at all, before any data or
    /// weights are supplied.
    ///
    /// The one check both [`Self::preview_transform`] and the apply run, so a
    /// preview never authorizes what the apply refuses. Only retarget depends
    /// on the handle (it needs a prepared score table); a compatible-data
    /// refresh is always available, and its schema check needs the new data.
    /// The remaining intents change the question or structure and are
    /// performed by preparing a new study, not applied to this handle.
    ///
    /// # Errors
    ///
    /// The reason-coded refusal the apply raises.
    pub fn transform_capability(
        &self,
        intent: antecedent_core::TransformIntent,
    ) -> Result<(), CausalError> {
        match intent {
            antecedent_core::TransformIntent::Retarget => self.retarget_score_table().map(|_| ()),
            _ => Ok(()),
        }
    }

    /// Estimate `E_Q[μ_a(X)]` from frozen scores. Does not refit or re-identify.
    ///
    /// `weights` must align with the score-table complete-case rows.
    /// `depends_on` is the declared parent set of `w`; it must be a subset of
    /// the certified adjustment set and must not name the treatment, an
    /// intervened coordinate, or a descendant.
    ///
    /// # Errors
    ///
    /// Missing score table, illegal `depends_on`, weight shape, or weighted
    /// overlap failure (support refusal).
    pub fn retarget(
        &self,
        weights: &[f64],
        depends_on: &[antecedent_core::VariableId],
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let _ = ctx;
        let table = self.retarget_score_table()?;
        let graph: Option<&dyn antecedent_estimate::DirectedAncestry> = self
            .analysis
            .graph
            .as_dag()
            .map(|g| g as _)
            .or_else(|| self.analysis.graph.as_admg().map(|g| g as _));
        let treatment = score_table_treatment_col(&self.analysis, table);
        let (out, overlap_failed) = antecedent_estimate::retarget(
            table,
            weights,
            depends_on,
            graph,
            treatment.as_deref(),
            None,
        )?;
        if overlap_failed {
            return Err(CausalError::Support {
                id: crate::support::SupportRefusal::Refused,
                message: antecedent_estimate::RetargetRefusal::WeightedOverlap.as_str(),
            });
        }
        let inference = table.inference(Some(weights))?;
        self.retarget_to_result(out, inference, weights, depends_on)
    }

    #[allow(clippy::float_cmp)] // Exact membership in binary intervention levels.
    fn retarget_to_result(
        &self,
        out: RetargetResult,
        inference: antecedent_estimate::scores::ScoreInference,
        weights: &[f64],
        depends_on: &[antecedent_core::VariableId],
    ) -> Result<StudyResult, CausalError> {
        let cache =
            self.analysis.identification_cache.as_ref().ok_or(CausalError::Unsupported {
                message: "retarget requires prepare-time identification",
            })?;
        let table = self
            .score_table
            .as_ref()
            .ok_or(CausalError::Unsupported { message: "missing frozen scores" })?;
        let n_thresholds = table.distinct_threshold_count();
        let quantile = match &self.analysis.query {
            CausalQuery::AverageEffect(q) => q
                .outcome_functional
                .quantile_level()
                .map(|tau| {
                    antecedent_estimate::quantile::quantile_contrast(table, Some(weights), tau)
                        .map(|q| (q.value, q.influence))
                })
                .transpose()?,
            CausalQuery::Response(q) => q
                .outcome_functional
                .quantile_level()
                .map(|tau| {
                    let arm = super::helpers::requested_joint_arm(q)?;
                    let q = antecedent_estimate::quantile::quantile_arm(
                        table,
                        Some(weights),
                        tau,
                        arm,
                    )?;
                    Ok::<_, CausalError>((q.value, q.influence))
                })
                .transpose()?,
            _ => None,
        };
        let (ate, se) = match &self.analysis.query {
            CausalQuery::AverageEffect(_) | CausalQuery::Response(_) if n_thresholds > 1 => {
                (f64::NAN, f64::NAN)
            }
            CausalQuery::AverageEffect(_) => {
                let c = out
                    .contrast
                    .as_ref()
                    .ok_or(CausalError::Unsupported { message: "missing arm contrast" })?;
                (c.value, c.se)
            }
            CausalQuery::Response(q) => {
                let arm = super::helpers::requested_joint_arm(q)?;
                let col = table
                    .columns
                    .iter()
                    .position(|c| c.arm == arm)
                    .ok_or(CausalError::Unsupported { message: "missing joint cell" })?;
                (out.summary.means[col], out.covariance.se(col))
            }
            _ => return Err(CausalError::Unsupported { message: "unsupported retarget query" }),
        };
        let cdf = self.score_table.as_ref().and_then(|t| exceedance_cdf_values(&out.summary, t));
        let mut estimate = EffectEstimate::new(
            ate,
            se,
            cache.identification.required_assumptions.clone(),
            OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: None },
        )
        .with_score_table(self.score_table.clone())
        .with_joint_covariance(Some(out.covariance.clone()))
        .with_exceedance_cdf(cdf)
        .with_monotone_rearranged(out.monotone_rearranged);
        estimate.score_inference = Some(inference);
        if let Some((value, influence)) = quantile.as_ref() {
            estimate.ate = *value;
            estimate.se_analytic =
                antecedent_estimate::joint_influence_covariance(&[influence], None)?.se(0);
            estimate.influence = Some(influence.clone().into());
        }
        let (treatment, outcome) = match &self.analysis.query {
            CausalQuery::AverageEffect(q) => (q.treatment, q.outcome),
            CausalQuery::Response(q) => q
                .functional
                .primary_pair()
                .ok_or(CausalError::Unsupported { message: "retarget response missing pair" })?,
            _ => {
                return Err(CausalError::Unsupported {
                    message: "retarget is licensed for AverageEffect and InterventionResponse",
                });
            }
        };
        let mut diagnostics = out.diagnostics;
        if quantile.is_some() {
            diagnostics.push(antecedent_core::Diagnostic::new(
                "estimate.functional.quantile", antecedent_core::DiagnosticKind::Scientific,
                antecedent_core::DiagnosticSeverity::Info,
                "weighted piecewise-linear CDF inversion conditional on the frozen grid; grid-selection uncertainty and interpolation bias are excluded",
            ));
        }
        if n_thresholds > 1 && quantile.is_none() {
            diagnostics.push(antecedent_core::Diagnostic::new(
                "estimate.functional.grid_scalar_cleared",
                antecedent_core::DiagnosticKind::Scientific,
                antecedent_core::DiagnosticSeverity::Info,
                "exceedance grids do not publish a first-threshold scalar ATE; use exceedance_cdf and the score table",
            ));
        }
        diagnostics.push(antecedent_core::Diagnostic::new(
            "estimate.aipw.crossfit_scores",
            antecedent_core::DiagnosticKind::Scientific,
            antecedent_core::DiagnosticSeverity::Info,
            "retarget averages the prepared cross-fitted φ table; it is not a residualized full-sample AIPW refit, so it can differ from the estimator's own point value under uniform weights",
        ));
        diagnostics.push(antecedent_core::Diagnostic::new(
            "retarget.selection_assumption", antecedent_core::DiagnosticKind::Scientific,
            antecedent_core::DiagnosticSeverity::Info,
            "weights are caller-declared fixed functions of certified covariates; inference assumes iid sampling, positivity, and nuisance convergence; selection or weight-estimation uncertainty is excluded",
        ));
        diagnostics.push(antecedent_core::Diagnostic::new(
            "exec.identify.cached",
            antecedent_core::DiagnosticKind::Execution,
            antecedent_core::DiagnosticSeverity::Info,
            "identification reused from the prepare-time cache",
        ));
        let mut result = super::helpers::assemble_result(super::helpers::AssembleArgs {
            logical: &self.plan.logical.record,
            physical: &self.plan.record,
            identification: cache.identification.clone(),
            estimand: cache.estimand.clone(),
            estimate,
            distribution: None,
            posterior: None,
            mediation: None,
            mediation_grid: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations: Vec::new(),
            diagnostics,
            provenance: antecedent_core::ProvenanceGraph::new(),
            treatment,
            outcome,
            wall_time_ns: 0,
            latency_mode: None,
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: None,
            bootstrap_replicates_ok: None,
            n_draws: None,
            cancelled: false,
            early_stopped: false,
            bayesian: matches!(self.analysis.inference, InferenceMode::Bayesian(_)),
        });
        result.certificate = Some(crate::result::AnalysisIdentification {
            identification: crate::Identification::Point {
                result: cache.identification.clone(),
                temporal_indexer: None,
                strategy: self
                    .plan
                    .logical
                    .record
                    .identifier
                    .as_deref()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(crate::strategy_table::DEFAULT_IDENTIFIER_ID),
                structure_version: self.analysis.graph.version(),
            },
            query: self.analysis.query.clone(),
            graph_class: self.analysis.graph.class(),
        });
        if antecedent_estimate::changes_target(weights) {
            let binding = self.row_weights_binding(weights, depends_on)?;
            if let Some(certificate) = &mut result.certificate {
                if let Some(target) = certificate.query.target_population_mut() {
                    *target = binding.population();
                }
            }
            result.row_weights = Some(binding);
        }
        if let CausalQuery::Response(q) = &self.analysis.query {
            result.response = Some(antecedent_core::CausalResponse {
                estimand: q.functional.clone(),
                identification_status: cache.identification.status,
                estimate: antecedent_core::ResponseIdentification::PointIdentified(
                    antecedent_core::ResponseValue::Scalar(result.estimate.ate),
                ),
                uncertainty: antecedent_core::ResponseUncertainty::Scalar {
                    standard_error: result.estimate.se_analytic,
                    lower: result.estimate.ate
                        - crate::result::reported_se_interval_z() * result.estimate.se_analytic,
                    upper: result.estimate.ate
                        + crate::result::reported_se_interval_z() * result.estimate.se_analytic,
                    level: 0.95,
                    interpretation: antecedent_core::IntervalInterpretation::Confidence,
                    draws: None,
                },
                support: antecedent_core::SupportReport {
                    status: antecedent_core::SupportStatus::Supported,
                    query_region: antecedent_core::SupportRegion {
                        minima: Arc::from([]),
                        maxima: Arc::from([]),
                    },
                    diagnostics: Vec::new(),
                    warnings: Vec::new(),
                    point_status: None,
                },
                assumptions: cache.identification.required_assumptions.clone(),
                provenance_id: Arc::from("estimate.cell.aipw.retarget"),
                horizon_identification: None,
                interaction_structurally_zero: false,
            });
        }
        result.rebind_interval(matches!(self.analysis.inference, InferenceMode::Bayesian(_)));
        result.support_status = self.analysis.support_status;
        result.structure_source = self.analysis.structure_source;
        let retargeted = result.retarget_population();
        result.executed_contract = Some(self.executed_contract(
            &self.analysis.data,
            self.analysis.refute,
            retargeted.as_ref(),
        )?);
        Ok(result)
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
        let shared = self
            .analysis
            .shared_batch_design
            .as_ref()
            .map(|s| s.rebind(data).map(Arc::new))
            .transpose()?;
        self.estimate_with_shared(data, shared, ctx)
    }

    pub(crate) fn estimate_with_shared(
        &self,
        data: &TabularData,
        shared: Option<Arc<super::batch::SharedBatchDesign>>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.ensure_schema_compatible(data)?;
        let mut click_analysis = self.analysis.clone();
        click_analysis.data = DataInput::Tabular(data.clone());
        click_analysis.interference =
            click_analysis.interference.as_ref().map(|spec| spec.bound_to(data)).transpose()?;
        click_analysis.shared_batch_design = shared;
        let mut result = click_analysis.execute_tabular(data, &self.plan, ctx)?;
        // `execute_tabular` bypasses `Study::execute_on`, which is where fresh runs
        // record which refutation reports are caller-attested. Without the names,
        // the claim would drop custom-validator evidence from `attested` and from
        // the claim identity.
        result.custom_validator_names = click_analysis
            .custom_validators
            .iter()
            .map(|validator| Arc::from(validator.name()))
            .collect();
        // Only a non-mean functional is read from the frozen scores; a mean click keeps
        // the estimator's own value, so refitting the cross-fit table would be discarded.
        let click_scores = if query_reads_score_table(&self.analysis.query) {
            click_analysis.prepare_score_table(ctx)?
        } else {
            None
        };
        overlay_prepared_score_functional(
            &self.analysis.query,
            click_scores.as_ref(),
            &mut result,
        )?;
        self.stamp(&click_analysis.data, result)
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
        self.transform_capability(antecedent_core::TransformIntent::CompatibleDataReplace)?;
        self.ensure_schema_compatible(&data)?;
        let mut refreshed = self.analysis.clone();
        refreshed.shared_batch_design = refreshed
            .shared_batch_design
            .as_ref()
            .map(|s| s.rebind(&data).map(Arc::new))
            .transpose()?;
        refreshed.interference =
            refreshed.interference.as_ref().map(|spec| spec.bound_to(&data)).transpose()?;
        refreshed.data = DataInput::Tabular(data);
        let mut result = refreshed.execute(&self.plan, ctx)?;
        let scores = refreshed.prepare_score_table(ctx)?;
        overlay_prepared_score_functional(&refreshed.query, scores.as_ref(), &mut result)?;
        self.replace_study(refreshed);
        self.score_table = scores;
        let data = self.analysis.data.clone();
        self.stamp(&data, result)
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
            return self.stamp_refuted(prior, data, suite, result);
        }
        let query = match &self.analysis.query {
            CausalQuery::AverageEffect(query) => query.clone(),
            CausalQuery::Response(response)
                if matches!(
                    response.functional,
                    antecedent_core::ResponseFunctional::InterventionResponse { .. }
                ) && prior.estimate.ate.is_finite() =>
            {
                AverageEffectQuery::binary_ate(prior.treatment, prior.outcome)
            }
            _ => {
                return Err(CausalError::Support {
                    id: crate::support::SupportRefusal::Refused,
                    message: "PreparedStudy::refute is licensed for AverageEffect and scalar \
                              InterventionResponse",
                });
            }
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
            super::helpers::mirror_refuted_evalue(&mut out.estimate, &out.refutations);
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
            return self.stamp_refuted(prior, data, suite, out);
        }
        let estimator = self.plan.logical.record.estimator.as_deref().unwrap_or(DEFAULT_ESTIMATOR);

        let (data_est, query_est, estimand_est) =
            project_for_ate_estimate(data, &query, &prior.estimand)?;

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
        super::helpers::mirror_refuted_evalue(&mut out.estimate, &out.refutations);
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
        self.stamp_refuted(prior, data, suite, out)
    }

    /// Stamp a second-click refute under the refute suite that produced its
    /// refutations. The estimate and the refutations must come from this
    /// handle on the same data snapshot; otherwise the result mixes two
    /// executions and carries no contract stamp (it cannot be exported).
    fn stamp_refuted(
        &self,
        prior: &StudyResult,
        data: &TabularData,
        suite: RefuteSuite,
        mut out: StudyResult,
    ) -> Result<StudyResult, CausalError> {
        let input = DataInput::Tabular(data.clone());
        let retargeted = prior.retarget_population();
        let population = retargeted.as_ref();
        let same_execution = match &prior.executed_contract {
            Some(stamp) => *stamp == self.executed_contract(&input, stamp.refute, population)?,
            None => false,
        };
        out.executed_contract = if same_execution {
            Some(self.executed_contract(&input, suite, population)?)
        } else {
            None
        };
        Ok(out)
    }

    /// Refuse data of a modality other than the one the plan was prepared for.
    fn ensure_modality(&self, requested: PreparedModality) -> Result<(), CausalError> {
        if self.modality == requested {
            return Ok(());
        }
        Err(CausalError::Compile {
            message: format!(
                "prepared {prepared} analysis requires {prepared} data; use {entry} \
                 (re-prepare to analyse {requested} data)",
                prepared = self.modality.label(),
                entry = self.modality.estimate_entry(),
                requested = requested.label(),
            ),
        })
    }

    /// Refuse a time index whose sampling regularity differs from prepare time.
    fn ensure_regularity(
        &self,
        regularity: &antecedent_data::SamplingRegularity,
    ) -> Result<(), CausalError> {
        if self.time_regularity.as_ref() == Some(regularity) {
            return Ok(());
        }
        Err(CausalError::Compile {
            message: format!(
                "prepared {} analysis requires the same time-index regularity as \
                 prepare-time data; re-prepare after a time-index change",
                self.modality.label()
            ),
        })
    }

    fn ensure_schema_compatible(&self, data: &TabularData) -> Result<(), CausalError> {
        self.ensure_modality(PreparedModality::Tabular)?;
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
        self.ensure_modality(PreparedModality::Series)?;
        if data.schema() != &self.schema {
            return Err(CausalError::Compile {
                message: "prepared temporal analysis requires the same schema \
                    (variable names, types, and order) as prepare-time data"
                    .into(),
            });
        }
        self.ensure_regularity(&data.time_index().regularity)
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
        let input = self.series_input(data.clone());
        let result = self.analysis.execute_on(&input, &self.plan, ctx)?;
        self.stamp(&input, result)
    }

    /// Re-estimate a prepared panel Pulse/Sustained analysis (no re-identify).
    ///
    /// Compatible new units and observations are admitted when schema and
    /// sampling regularity match. Incompatible schema, time regularity, or an
    /// empty panel is refused and the retained handle is unchanged.
    ///
    /// # Errors
    ///
    /// Schema / regularity mismatch, empty panel, or estimation failures.
    pub fn estimate_panel(
        &self,
        data: &PanelData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.ensure_panel_compatible(data)?;
        let input = DataInput::Panel(data.clone());
        let result = self.analysis.execute_on(&input, &self.plan, ctx)?;
        self.stamp(&input, result)
    }

    /// Replace retained panel data and re-estimate without re-identifying.
    ///
    /// # Errors
    ///
    /// Same refusals as [`Self::estimate_panel`].
    pub fn refresh_panel(
        &mut self,
        data: PanelData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.transform_capability(antecedent_core::TransformIntent::CompatibleDataReplace)?;
        self.ensure_panel_compatible(&data)?;
        let mut refreshed = self.analysis.clone();
        refreshed.data = DataInput::Panel(data);
        let result = refreshed.execute(&self.plan, ctx)?;
        self.replace_study(refreshed);
        let data = self.analysis.data.clone();
        self.stamp(&data, result)
    }

    /// Re-estimate a prepared multi-environment temporal analysis (no re-identify).
    ///
    /// # Errors
    ///
    /// Schema / regularity mismatch, or estimation failures.
    pub fn estimate_multi_env(
        &self,
        data: &antecedent_data::MultiEnvironmentData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.ensure_multi_env_compatible(data)?;
        let input = DataInput::MultiEnv(data.clone());
        let result = self.analysis.execute_on(&input, &self.plan, ctx)?;
        self.stamp(&input, result)
    }

    /// Replace retained multi-environment data and re-estimate without re-identifying.
    ///
    /// # Errors
    ///
    /// Same refusals as [`Self::estimate_multi_env`].
    pub fn refresh_multi_env(
        &mut self,
        data: antecedent_data::MultiEnvironmentData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.transform_capability(antecedent_core::TransformIntent::CompatibleDataReplace)?;
        self.ensure_multi_env_compatible(&data)?;
        let mut refreshed = self.analysis.clone();
        refreshed.data = DataInput::MultiEnv(data);
        let result = refreshed.execute(&self.plan, ctx)?;
        self.replace_study(refreshed);
        let data = self.analysis.data.clone();
        self.stamp(&data, result)
    }

    fn ensure_multi_env_compatible(
        &self,
        data: &antecedent_data::MultiEnvironmentData,
    ) -> Result<(), CausalError> {
        self.ensure_modality(PreparedModality::MultiEnv)?;
        if data.schema() != &self.schema {
            return Err(CausalError::Compile {
                message: "prepared multi-env analysis requires the same schema \
                    (variable names, types, and order) as prepare-time data"
                    .into(),
            });
        }
        for env in data.environments() {
            self.ensure_regularity(&env.time_index().regularity)?;
        }
        Ok(())
    }

    fn ensure_panel_compatible(&self, data: &PanelData) -> Result<(), CausalError> {
        self.ensure_modality(PreparedModality::Panel)?;
        if data.schema() != &self.schema {
            return Err(CausalError::Compile {
                message: "prepared panel analysis requires the same schema \
                    (variable names, types, and order) as prepare-time data"
                    .into(),
            });
        }
        if data.unit_count() == 0 {
            return Err(CausalError::Compile {
                message: "prepared panel refresh refuses an empty panel; prior state is retained"
                    .into(),
            });
        }
        for unit in data.units() {
            self.ensure_regularity(&unit.series.time_index().regularity)?;
        }
        Ok(())
    }

    /// Estimate using the data retained by preparation or the latest successful refresh.
    ///
    /// Uses the existing modality-specific executor and preserves identification caches.
    ///
    /// # Errors
    ///
    /// Execution failures, or an unsupported retained data modality.
    pub fn estimate_retained(&self, ctx: &ExecutionContext) -> Result<StudyResult, CausalError> {
        match &self.analysis.data {
            DataInput::Tabular(data) => self.estimate(data, ctx),
            DataInput::Temporal(data) | DataInput::Event(data) => self.estimate_series(data, ctx),
            DataInput::Panel(data) => self.estimate_panel(data, ctx),
            DataInput::MultiEnv(data) => self.estimate_multi_env(data, ctx),
        }
    }

    /// Replace retained series and re-estimate.
    ///
    /// The handle is updated only when estimation succeeds; a refused or
    /// failed refresh leaves the retained series unchanged.
    ///
    /// # Errors
    ///
    /// Schema / time-index mismatch, or estimation failures.
    pub fn refresh_series(
        &mut self,
        data: TimeSeriesData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.transform_capability(antecedent_core::TransformIntent::CompatibleDataReplace)?;
        self.ensure_series_compatible(&data)?;
        let mut refreshed = self.analysis.clone();
        refreshed.data = self.series_input(data);
        let result = refreshed.execute(&self.plan, ctx)?;
        self.replace_study(refreshed);
        let data = self.analysis.data.clone();
        self.stamp(&data, result)
    }
}

/// Sampling regularity shared by every unit of a prepare-time panel.
///
/// Refresh requires every unit to match the frozen regularity, so prepare
/// refuses a panel whose units already disagree rather than freezing one
/// unit's grid and then refusing the same data on refresh.
fn multi_env_regularity(
    multi: &antecedent_data::MultiEnvironmentData,
) -> Result<antecedent_data::SamplingRegularity, CausalError> {
    let first = multi
        .environment(0)
        .map_err(|err| CausalError::Compile { message: err.to_string() })?
        .time_index()
        .regularity
        .clone();
    if multi.environments().iter().any(|env| env.time_index().regularity != first) {
        return Err(CausalError::Compile {
            message: "PreparedStudy requires every multi-env series to share one time-index \
                      regularity; align the environments before preparing"
                .into(),
        });
    }
    Ok(first)
}

impl Study {
    /// Compile once into a durable [`PreparedStudy`] for re-estimate-many.
    ///
    /// Supports:
    /// - tabular [`CausalQuery::AverageEffect`] on a supplied static graph
    ///   ([`GraphClass::Dag`], [`GraphClass::Cpdag`], [`GraphClass::Pag`], or
    ///   [`GraphClass::Admg`])
    /// - tabular [`CausalQuery::AverageEffect`], [`CausalQuery::ConditionalEffect`],
    ///   and static [`CausalQuery::Response`] on a supplied DAG graph posterior
    /// - tabular [`CausalQuery::Response`] and [`CausalQuery::ConditionalEffect`]
    ///   on a supplied [`GraphClass::Dag`], [`GraphClass::Cpdag`],
    ///   [`GraphClass::Pag`], or [`GraphClass::Admg`] (joint Response also on a
    ///   CoDetermined tier closure)
    /// - tabular [`CausalQuery::Distribution`] on a supplied [`GraphClass::Dag`]
    ///   or [`GraphClass::Admg`] (ADMG: unconditional, validation none)
    /// - tabular [`CausalQuery::PathSpecific`], static [`CausalQuery::Mediation`],
    ///   and [`CausalQuery::Counterfactual`] on a supplied [`GraphClass::Dag`]
    /// - series temporal [`CausalQuery::Response`] on a supplied
    ///   [`GraphClass::TemporalDag`], [`GraphClass::TemporalCpdag`], or
    ///   [`GraphClass::TemporalPag`]
    /// - series [`CausalQuery::TemporalEffect`] (Pulse / single-step Sustained)
    ///   on a supplied [`GraphClass::TemporalDag`], [`GraphClass::TemporalCpdag`],
    ///   or [`GraphClass::TemporalPag`]
    /// - series [`CausalQuery::TemporalEffect`] (Pulse / single-step or
    ///   multi-step Sustained) on a supplied DBN graph posterior
    /// - series temporal [`CausalQuery::Response`] on a supplied DBN graph
    ///   posterior (TemporalDag atoms; MeanCurve none, one-coordinate
    ///   InterventionResponse none/cheap/full)
    /// - scalar-horizon series [`CausalQuery::Mediation`] (`TemporalMediationEffect`) on a
    ///   supplied [`GraphClass::TemporalDag`] or [`GraphClass::TemporalCpdag`]
    ///   (horizon-specific `I(h)`)
    /// - scalar-horizon series [`CausalQuery::Mediation`] (`TemporalMediationEffect`) on a
    ///   supplied DBN graph posterior (per-atom `I(h)`, unidentified mass retained)
    /// - panel [`CausalQuery::TemporalEffect`] (Pulse / Sustained) on a supplied
    ///   [`GraphClass::TemporalDag`]; every unit must share one time-index regularity
    ///
    /// The handle records the prepare-time data modality (tabular, series, or
    /// panel) and its estimate / refresh entry points refuse other modalities.
    /// Discovery inputs and review-required compiles are refused.
    ///
    /// # Errors
    ///
    /// Unsupported combination, compile failure, or review-required plan.
    pub fn prepare(&self, ctx: &ExecutionContext) -> Result<PreparedStudy, CausalError> {
        ensure_prepared_supported(self)?;
        let plan = self.compile(ctx)?;
        let (schema, modality, time_regularity) = match &self.data {
            DataInput::Tabular(data) => (data.schema().clone(), PreparedModality::Tabular, None),
            DataInput::Temporal(data) | DataInput::Event(data) => (
                data.schema().clone(),
                PreparedModality::Series,
                Some(data.time_index().regularity.clone()),
            ),
            DataInput::Panel(panel) => (
                panel.schema().clone(),
                PreparedModality::Panel,
                Some(super::builder::panel_shared_regularity(panel)?),
            ),
            DataInput::MultiEnv(multi) => (
                multi.schema().clone(),
                PreparedModality::MultiEnv,
                Some(multi_env_regularity(multi)?),
            ),
        };
        let mut analysis = self.clone();
        match (&self.data, &self.query, self.graph_posterior.as_ref()) {
            (DataInput::Tabular(_), CausalQuery::AverageEffect(query), Some(posterior)) => {
                analysis.graph_posterior_identification_cache = Some(Arc::new(
                    build_graph_posterior_identification_cache(posterior, query, ctx)?,
                ));
            }
            (DataInput::Tabular(_), CausalQuery::ConditionalEffect(query), Some(posterior)) => {
                analysis.graph_posterior_identification_cache = Some(Arc::new(
                    build_graph_posterior_identification_cache(posterior, &query.inner, ctx)?,
                ));
            }
            (DataInput::Tabular(_), CausalQuery::Response(query), Some(posterior))
                if !query.is_temporal() =>
            {
                analysis.graph_posterior_identification_cache = Some(Arc::new(
                    if matches!(
                        posterior.atom_kind,
                        antecedent_discovery::GraphPosteriorAtomKind::Admg
                    ) {
                        build_admg_graph_posterior_response_identification_cache(
                            posterior, query, ctx,
                        )?
                    } else {
                        let (treatment, outcome) =
                            query.functional.primary_pair().ok_or_else(|| {
                                CausalError::Compile {
                                    message: "response query has no treatment/outcome pair".into(),
                                }
                            })?;
                        build_graph_posterior_identification_cache(
                            posterior,
                            &AverageEffectQuery::binary_ate(treatment, outcome),
                            ctx,
                        )?
                    },
                ));
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(query),
                Some(posterior),
            ) => {
                let variables: Vec<_> =
                    data.schema().variables().iter().map(|variable| variable.id).collect();
                if matches!(
                    posterior.atom_kind,
                    antecedent_discovery::GraphPosteriorAtomKind::Cpdag
                        | antecedent_discovery::GraphPosteriorAtomKind::Pag
                ) {
                    analysis.temporal_class_posterior_identification_cache =
                        Some(Arc::new(build_temporal_class_posterior_identification_cache(
                            posterior,
                            &variables,
                            query,
                            analysis.max_completions,
                            ctx,
                        )?));
                } else {
                    analysis.dbn_posterior_identification_cache =
                        Some(Arc::new(build_dbn_posterior_identification_cache(
                            posterior, &variables, query, ctx,
                        )?));
                }
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::Mediation(query),
                Some(posterior),
            ) => {
                let variables: Vec<_> =
                    data.schema().variables().iter().map(|variable| variable.id).collect();
                if matches!(
                    posterior.atom_kind,
                    antecedent_discovery::GraphPosteriorAtomKind::Cpdag
                        | antecedent_discovery::GraphPosteriorAtomKind::Pag
                ) {
                    let horizon =
                        query.horizons.first().copied().ok_or_else(|| CausalError::Compile {
                            message: "temporal class graph-posterior mediation requires a horizon"
                                .into(),
                        })?;
                    let mut witness =
                        TemporalEffectQuery::pulse(query.treatment, query.outcome, 1.0);
                    witness.horizon_steps = horizon;
                    analysis.temporal_class_posterior_identification_cache =
                        Some(Arc::new(build_temporal_class_posterior_identification_cache(
                            posterior,
                            &variables,
                            &witness,
                            analysis.max_completions,
                            ctx,
                        )?));
                } else {
                    analysis.dbn_posterior_identification_cache =
                        Some(Arc::new(build_dbn_posterior_mediation_identification_cache(
                            posterior, &variables, query, ctx,
                        )?));
                }
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::Response(query),
                Some(posterior),
            ) if query.is_temporal() => {
                super::execute::dbn_posterior_response_supported(query)?;
                let variables: Vec<_> =
                    data.schema().variables().iter().map(|variable| variable.id).collect();
                if matches!(
                    posterior.atom_kind,
                    antecedent_discovery::GraphPosteriorAtomKind::Cpdag
                        | antecedent_discovery::GraphPosteriorAtomKind::Pag
                ) {
                    let (treatment, outcome) =
                        query.functional.primary_pair().ok_or_else(|| CausalError::Compile {
                            message: "temporal class graph-posterior response has no \
                                      treatment/outcome pair"
                                .into(),
                        })?;
                    let horizon = query
                        .temporal
                        .as_ref()
                        .and_then(|spec| spec.horizons.first())
                        .copied()
                        .unwrap_or(1);
                    let mut witness = TemporalEffectQuery::pulse(treatment, outcome, 1.0);
                    witness.horizon_steps = horizon;
                    if let Some(temporal) = query.temporal.as_ref() {
                        witness.policy = temporal.policy.clone();
                        witness.max_history_lag = temporal.max_history_lag;
                    }
                    let cache = build_temporal_class_posterior_identification_cache(
                        posterior,
                        &variables,
                        &witness,
                        analysis.max_completions,
                        ctx,
                    )?;
                    if let Some(atom) = cache.class_atoms.first() {
                        analysis.temporal_class_identification_cache =
                            Some(Arc::new(CachedTemporalClassIdentification {
                                envelope: atom.envelope.clone(),
                                by_horizon: vec![(horizon, atom.envelope.clone())],
                            }));
                    }
                    analysis.temporal_class_posterior_identification_cache = Some(Arc::new(cache));
                } else {
                    let estimator_id = if matches!(self.inference, InferenceMode::Bayesian(_)) {
                        crate::strategy_table::EstimatorId::TemporalResponseBayesian
                    } else {
                        crate::strategy_table::EstimatorId::TemporalResponseGcomp
                    };
                    analysis.dbn_posterior_identification_cache =
                        Some(Arc::new(build_dbn_posterior_response_identification_cache(
                            posterior,
                            &variables,
                            query,
                            estimator_id,
                            ctx,
                        )?));
                }
            }
            (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::Mediation(_), None) => {
                analysis.temporal_identification_cache =
                    self.prepare_temporal_mediation_identification()?.map(Arc::new);
                analysis.temporal_class_identification_cache =
                    self.prepare_temporal_class_identification()?.map(Arc::new);
            }
            (DataInput::Tabular(_), CausalQuery::Transport(query), None) => {
                let diagram = self.selection_diagram.as_ref().ok_or(CausalError::Unsupported {
                    message: "TransportQuery prepare requires a selection diagram",
                })?;
                analysis.transport_identification_cache =
                    Some(Arc::new(super::execute::live_transport_identification(diagram, query)?));
            }
            (DataInput::Tabular(_), _, None) => {
                analysis.identification_cache =
                    self.prepare_static_identification(&plan)?.map(Arc::new);
                analysis.pag_identification_cache =
                    self.prepare_pag_identification(&plan)?.map(Arc::new);
                analysis.cpdag_identification_cache =
                    self.prepare_cpdag_identification(&plan)?.map(Arc::new);
            }
            (
                DataInput::Panel(_),
                CausalQuery::TemporalEffect(_) | CausalQuery::Response(_),
                None,
            ) => {
                if analysis.graph.class().is_incomplete_temporal() {
                    analysis.temporal_class_identification_cache =
                        self.prepare_temporal_class_identification()?.map(Arc::new);
                } else {
                    analysis.temporal_identification_cache =
                        self.prepare_temporal_identification()?.map(Arc::new);
                }
            }
            (_, _, None) => {
                analysis.temporal_identification_cache =
                    self.prepare_temporal_identification()?.map(Arc::new);
                analysis.temporal_class_identification_cache =
                    self.prepare_temporal_class_identification()?.map(Arc::new);
            }
            (_, _, Some(_)) => {
                // `ensure_prepared_supported` already refuses this coordinate; a
                // library must still answer with a typed refusal, never a panic.
                return Err(CausalError::Support {
                    id: crate::support::SupportRefusal::Refused,
                    message: "graph_posterior on the prepared handle is licensed only for \
                        tabular AverageEffect/Response/ConditionalEffect and series \
                        TemporalEffect, TemporalMediationEffect, or TemporalDag Response",
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
            || analysis.cpdag_identification_cache.is_some()
            || analysis.temporal_identification_cache.is_some()
            || analysis.temporal_class_identification_cache.is_some()
            || analysis.transport_identification_cache.is_some()
        {
            super::execute::report_identify_compute(ctx);
        }
        let score_table = analysis.prepare_score_table(ctx)?;
        Ok(PreparedStudy {
            state: SampledPreparedState {
                analysis,
                program_cache: std::sync::OnceLock::new(),
                plan,
                schema,
                modality,
                time_regularity,
                score_table,
            },
        })
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
            identify_static_query_with_rd, select_claim, select_estimand,
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
        if matches!(
            self.query,
            CausalQuery::AnomalyAttribution(_) | CausalQuery::ChangeAttribution(_)
        ) {
            let (treatment, outcome) = super::execute::gcm_query_vars(&self.query)?;
            let (identification, estimand) = super::execute::parametric_scm_identification(
                self.query.clone(),
                treatment,
                outcome,
            );
            return Ok(Some(CachedStaticIdentification { identification, estimand }));
        }
        let identifier = plan.logical.record.identifier.as_deref().unwrap_or(DEFAULT_IDENTIFIER);
        let estimator = plan.logical.record.estimator.as_deref().unwrap_or(DEFAULT_ESTIMATOR);
        let identifier_id: IdentifierId = identifier.parse()?;
        let estimator_id: EstimatorId = estimator.parse()?;
        if matches!(estimator_id, EstimatorId::RdSharp | EstimatorId::BayesianRdLocalLinear) {
            return Ok(None);
        }
        match &self.query {
            CausalQuery::AverageEffect(query) => {
                if let Some(background) = &self.tiered {
                    let identification = antecedent_identify::identify_tiered(background, query)?;
                    let estimand = identification.estimands.first().cloned().ok_or_else(|| {
                        CausalError::Compile {
                            message: "tiered identification returned no estimand".into(),
                        }
                    })?;
                    return Ok(Some(CachedStaticIdentification { identification, estimand }));
                }
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
                    GraphClass::Admg => plan.static_graph().cloned(),
                    GraphClass::Cpdag => return Ok(None),
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
                let (identification, estimand) = select_claim(identification, estimator_id)?;
                Ok(Some(CachedStaticIdentification { identification, estimand }))
            }
            CausalQuery::Response(query) => {
                if let Some(background) = &self.tiered {
                    let schema = match &self.data {
                        DataInput::Tabular(data) => data.schema(),
                        _ => {
                            return Err(CausalError::Unsupported {
                                message: "CoDetermined joint prepare requires tabular data",
                            });
                        }
                    };
                    let identification = match self.graph.as_admg() {
                        Some(admg) => {
                            antecedent_identify::identify_tiered_joint_on(background, admg, query)?
                        }
                        None => {
                            antecedent_identify::identify_tiered_joint(background, schema, query)?
                        }
                    };
                    let estimand = identification.estimands.first().cloned().ok_or(
                        CausalError::Unsupported {
                            message: antecedent_identify::TIERED_JOINT_ADJUSTMENT_REFUSE,
                        },
                    )?;
                    return Ok(Some(CachedStaticIdentification { identification, estimand }));
                }
                if self.graph.class() == GraphClass::Admg
                    && self.graph.as_admg().is_some_and(super::execute::admg_has_bidirected)
                {
                    use crate::strategy_table::identify_admg_query;

                    let admg = self.graph.as_admg().ok_or_else(|| CausalError::Compile {
                        message: "ADMG prepare missing supplied graph".into(),
                    })?;
                    // `execute_admg_response` reuses this cache verbatim as the *first grid
                    // level's* claim for a MeanCurve (it only re-identifies per level from the
                    // second level on, mirroring the graph-posterior ADMG response cache). A
                    // MeanCurve identified whole produces one general.id estimand per grid
                    // level, and `select_estimand` then has no unique estimator match to pick
                    // among them (they all report the same method). Cache the first level's
                    // InterventionResponse claim instead, which is what downstream code
                    // actually consumes and — being a single intervention level — is exactly
                    // what `select_estimand` can disambiguate.
                    let causal_query = match &query.functional {
                        antecedent_core::ResponseFunctional::MeanCurve { outcome, treatment } => {
                            let first_level = treatment
                                .grid
                                .values()
                                .map_err(|e| CausalError::Compile { message: e.to_string() })?
                                .into_iter()
                                .next()
                                .ok_or_else(|| CausalError::Compile {
                                    message: "MeanCurve response requires a non-empty evaluation \
                                              grid"
                                        .into(),
                                })?;
                            let mut level_query = query.clone();
                            level_query.functional =
                                antecedent_core::ResponseFunctional::InterventionResponse {
                                    outcome: *outcome,
                                    interventions: Arc::from([Intervention::set(
                                        treatment.variable,
                                        Value::f64(first_level),
                                    )]),
                                };
                            CausalQuery::Response(level_query)
                        }
                        _ => CausalQuery::Response(query.clone()),
                    };
                    let identification = identify_admg_query(identifier_id, admg, &causal_query)?;
                    let estimand = select_estimand(&identification, estimator_id)?;
                    return Ok(Some(CachedStaticIdentification { identification, estimand }));
                }
                let graph = match (self.graph.as_dag(), plan.static_graph()) {
                    (Some(dag), _) => Some(dag.clone()),
                    (None, Some(dag)) if self.graph.class() == GraphClass::Admg => {
                        Some(dag.clone())
                    }
                    _ => None,
                };
                let Some(graph) = graph else {
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
                let (identification, estimand) = select_claim(identification, estimator_id)?;
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
                // Mirrors `execute_distribution`.
                use super::execute::DistributionGraph;
                use crate::strategy_table::{
                    DEFAULT_DISTRIBUTION_ESTIMATOR, DEFAULT_DISTRIBUTION_IDENTIFIER,
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
                let graph = if let Some(admg) = self.graph.as_admg() {
                    DistributionGraph::Admg(admg)
                } else if let Some(dag) = self.graph.as_dag() {
                    DistributionGraph::Dag(dag)
                } else {
                    return Ok(None);
                };
                let identification = graph.identify(identifier_id, query)?;
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

        let Some(query) = self.envelope_witness_ate()? else {
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
        let envelope = if let CausalQuery::Response(response) = &self.query {
            crate::strategy_table::identify_pag_response(identifier_id, pag, response)?
        } else {
            identify_pag(identifier_id, pag, &query)?
        };
        let identification =
            super::execute::envelope_to_identification_result_for(&envelope, self.query.clone());
        Ok(Some(CachedPagIdentification { envelope, identification }))
    }

    /// Compute the MEC envelope once for a supplied CPDAG.
    fn prepare_cpdag_identification(
        &self,
        plan: &PhysicalExecutionPlan,
    ) -> Result<Option<CachedCpdagIdentification>, CausalError> {
        use crate::strategy_table::{DEFAULT_PAG_IDENTIFIER, IdentifierId, identify_cpdag};

        let Some(query) = self.envelope_witness_ate()? else {
            return Ok(None);
        };
        if self.graph.class() != GraphClass::Cpdag {
            return Ok(None);
        }
        let cpdag = self.graph.as_cpdag().ok_or_else(|| CausalError::Compile {
            message: "CPDAG prepare missing supplied graph".into(),
        })?;
        let identifier =
            plan.logical.record.identifier.as_deref().unwrap_or(DEFAULT_PAG_IDENTIFIER);
        let identifier_id: IdentifierId = identifier.parse()?;
        let envelope = if let CausalQuery::Response(response) = &self.query {
            crate::strategy_table::identify_cpdag_response(identifier_id, cpdag, response)?
        } else {
            identify_cpdag(identifier_id, cpdag, &query)?
        };
        let identification =
            super::execute::envelope_to_identification_result_for(&envelope, self.query.clone());
        Ok(Some(CachedCpdagIdentification { envelope, identification }))
    }

    fn envelope_witness_ate(&self) -> Result<Option<AverageEffectQuery>, CausalError> {
        match &self.query {
            CausalQuery::AverageEffect(query) => Ok(Some(query.clone())),
            CausalQuery::Response(query)
                if super::execute::class_aware_response_supported(query) =>
            {
                Ok(Some(super::execute::response_witness_ate(query)?))
            }
            CausalQuery::ConditionalEffect(query) => Ok(Some(query.inner.clone())),
            _ => Ok(None),
        }
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
                let schedule = sequence_identification_schedule(query)?;
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
                    schedule.as_deref(),
                    single_step_dose(query)?,
                )?))
            }
            CausalQuery::TemporalEffect(query) => {
                let id_res = TemporalBackdoorIdentifier::new()
                    .with_parent_adjustment_fallback()
                    .identify_temporal(graph, query)
                    .map_err(CausalError::from)?;
                let estimand = select_estimand(
                    &id_res.result,
                    if query.is_multi_step_sustained() {
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

    fn prepare_temporal_class_identification(
        &self,
    ) -> Result<Option<CachedTemporalClassIdentification>, CausalError> {
        use crate::strategy_table::{DEFAULT_PAG_IDENTIFIER, IdentifierId};

        if !matches!(self.graph.class(), GraphClass::TemporalCpdag | GraphClass::TemporalPag) {
            return Ok(None);
        }
        let query = match &self.query {
            CausalQuery::TemporalEffect(query) => query.clone(),
            CausalQuery::Response(response) => {
                let temporal = response.temporal.as_ref().ok_or_else(|| CausalError::Compile {
                    message: "temporal class response prepare requires TemporalResponseSpec".into(),
                })?;
                let (treatment, outcome) =
                    response.functional.primary_pair().ok_or_else(|| CausalError::Compile {
                        message: "temporal class response has no treatment/outcome pair".into(),
                    })?;
                TemporalEffectQuery {
                    treatment,
                    outcome,
                    policy: temporal.policy.clone(),
                    control: Intervention::set(treatment, Value::f64(0.0)),
                    active: Intervention::set(treatment, Value::f64(1.0)),
                    horizon_steps: temporal.horizons.first().copied().unwrap_or(1),
                    max_history_lag: temporal.max_history_lag,
                    target_population: response.target_population.clone(),
                }
            }
            CausalQuery::Mediation(mediation)
                if self.graph.class() == GraphClass::TemporalCpdag =>
            {
                let mut witness =
                    TemporalEffectQuery::pulse(mediation.treatment, mediation.outcome, 1.0);
                witness.horizon_steps = mediation.horizons.first().copied().unwrap_or(1);
                witness
            }
            _ => return Ok(None),
        };
        let identifier = self.identifier.map_or(DEFAULT_PAG_IDENTIFIER, |id| id.as_str());
        let identifier_id: IdentifierId = identifier.parse()?;
        let mut bundle = self.identify_temporal_class(identifier_id, &query)?;
        let horizons: &[u32] = match &self.query {
            CausalQuery::Response(response) => {
                &response.temporal.as_ref().expect("temporal query").horizons
            }
            CausalQuery::Mediation(mediation) => &mediation.horizons,
            _ => &[],
        };
        for &horizon in horizons {
            let mut qh = query.clone();
            qh.horizon_steps = horizon;
            let identified = if horizon == query.horizon_steps {
                bundle.envelope.clone()
            } else {
                self.identify_temporal_class(identifier_id, &qh)?.envelope
            };
            bundle.by_horizon.push((horizon, identified));
        }
        Ok(Some(bundle))
    }

    /// One `I(h)` per requested mediation horizon (path-product + that horizon's backdoor `Z`).
    fn prepare_temporal_mediation_identification(
        &self,
    ) -> Result<Option<CachedTemporalIdentification>, CausalError> {
        use crate::strategy_table::EstimatorId;
        let CausalQuery::Mediation(query) = &self.query else {
            return Ok(None);
        };
        if self.graph.class() != GraphClass::TemporalDag {
            return Ok(None);
        }
        let graph = self.graph.as_temporal_dag().ok_or_else(|| CausalError::Compile {
            message: "temporal mediation prepare requires TemporalDag".into(),
        })?;
        Ok(Some(identify_temporal_mediation_horizons(
            graph,
            query,
            if matches!(self.inference, InferenceMode::Bayesian(_)) {
                EstimatorId::BayesianTemporalMediation
            } else {
                EstimatorId::TemporalMediation
            },
        )?))
    }

    /// Cross-fitted AIPW scores for retarget / exceedance / joint cells.
    pub(crate) fn prepare_score_table(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Option<ScoreTable>, CausalError> {
        let fold_seed = ctx.rng.master_seed();
        let DataInput::Tabular(data) = &self.data else {
            return Ok(None);
        };
        // Unknown is two canonical sets / GraphDependent / no single Z.
        // One score table would collapse the envelope; retarget refuses.
        if self
            .tiered
            .as_ref()
            .is_some_and(|b| b.within_tier == antecedent_graph::WithinTier::Unknown)
            || !matches!(self.inference, InferenceMode::Frequentist)
        {
            return Ok(None);
        }
        match &self.query {
            CausalQuery::AverageEffect(query) => {
                // Score artifacts are an explicit AIPW execution contract; do not
                // silently fit a second estimator for linear/IV/matching plans.
                if self.estimator != Some(crate::strategy_table::EstimatorId::Aipw)
                    || !matches!(query.target_population, TargetPopulation::AllObserved)
                {
                    return Ok(None);
                }
                let Some(cache) = self.identification_cache.as_ref() else {
                    return Ok(None);
                };
                let method = cache.estimand.method.as_ref();
                if !(method.contains("adjustment")
                    || method.contains("backdoor")
                    || method.starts_with("tiered."))
                {
                    return Ok(None);
                }
                let mut est = match self.estimator_spec.as_ref() {
                    Some(crate::estimator_spec::EstimatorSpec::Aipw(config)) => *config.clone(),
                    _ => AipwAte::new(),
                };
                if let Some(overlap) = self.overlap_policy {
                    est.overlap = overlap;
                }
                if matches!(est.overlap, OverlapPolicy::RequireDiagnostics { trim: Some(_), .. })
                    || est.se_kind != antecedent_estimate::AnalyticSeKind::Homoskedastic
                {
                    // Frozen cross-fitted scores are untrimmed iid influence
                    // values. A mean estimate under trimming or a non-iid SE is
                    // still estimated by the configured AIPW; it keeps no score
                    // table, so a retarget refuses with `score_table_unavailable`
                    // instead of reweighting scores of a different construction.
                    // A quantile or exceedance functional is estimated from the
                    // scores and has no such fallback.
                    if query.outcome_functional.is_mean() {
                        return Ok(None);
                    }
                    return Err(crate::unsupported_reason!(
                        "option_not_applicable",
                        "prepared AIPW functional scores require iid inference without \
                         propensity trimming"
                    ));
                }
                let estimand = cache.estimand.clone();
                let mut problem = est.prepare(data, &estimand, query)?;
                problem.fold_seed = fold_seed;
                if let Some(shared) = self.shared_batch_design.as_ref() {
                    shared.apply_to_propensity(&mut problem)?;
                }
                let table = if query.outcome_functional.quantile_level().is_some() {
                    let grid = antecedent_estimate::empirical_threshold_grid(&problem.outcome, 19)?;
                    antecedent_estimate::build_binary_scores(
                        &problem,
                        query.treatment,
                        &grid.into_iter().map(Some).collect::<Vec<_>>(),
                        antecedent_estimate::DEFAULT_AIPW_FOLDS,
                        &est.glm_options,
                        est.backend,
                    )?
                } else {
                    crossfit_binary_scores(
                        &problem,
                        query,
                        antecedent_estimate::DEFAULT_AIPW_FOLDS,
                        &est.glm_options,
                        est.backend,
                    )?
                };
                Ok(Some(table))
            }
            CausalQuery::Response(query) => {
                if self.estimator != Some(crate::strategy_table::EstimatorId::CellAipw) {
                    return Ok(None);
                }
                let Some(cache) = self.identification_cache.as_ref() else {
                    return Ok(None);
                };
                let antecedent_core::ResponseFunctional::InterventionResponse {
                    outcome,
                    interventions,
                } = &query.functional
                else {
                    return Ok(None);
                };
                let treatments = discrete_set_treatments(interventions);
                if treatments.len() < 2 {
                    return Ok(None);
                }
                let est = CellSaturatedAipw::new().with_fold_seed(fold_seed);
                let continuous = self.continuous_cell.as_ref().map(|(variable, grid)| {
                    antecedent_estimate::ContinuousCellSpec { variable: *variable, grid }
                });
                // Folds are deliberately not shared here: `fit_scores_with_assignment`
                // leaves them unset so the cell path draws its own cell-stratified
                // `crossfit_fold_plan` (keyed by `est.fold_seed`), reproducing the solo
                // fold plan for this query's own cells bit-for-bit instead of a private,
                // unstratified batch shuffle. See `SharedBatchDesign` docs.
                let fold_ids: Option<Vec<u32>> = None;
                let design = match self.shared_batch_design.as_ref() {
                    Some(shared) => {
                        let mut ids: Vec<_> = treatments
                            .iter()
                            .copied()
                            .chain(std::iter::once(*outcome))
                            .chain(cache.estimand.adjustment_set.iter().copied())
                            .collect();
                        if let Some((variable, _)) = self.continuous_cell.as_ref() {
                            ids.push(*variable);
                        }
                        let ids = data.complete_case_mask(&ids);
                        let row_index = match ids {
                            Ok(mask) => mask
                                .iter()
                                .enumerate()
                                .filter_map(|(i, &keep)| {
                                    keep.then_some(u32::try_from(i).unwrap_or(u32::MAX))
                                })
                                .collect::<Vec<_>>(),
                            Err(_) => Vec::new(),
                        };
                        if row_index.is_empty() {
                            None
                        } else {
                            shared.design_for(&cache.estimand.adjustment_set, &row_index)?
                        }
                    }
                    None => None,
                };
                let table = est.fit_scores_with_assignment(
                    data,
                    &treatments,
                    *outcome,
                    &cache.estimand.adjustment_set,
                    &query.outcome_functional,
                    continuous,
                    fold_ids.as_deref(),
                    design.as_deref(),
                )?;
                Ok(Some(table))
            }
            _ => Ok(None),
        }
    }
}

fn discrete_set_treatments(interventions: &[Intervention]) -> Vec<antecedent_core::VariableId> {
    interventions
        .iter()
        .filter_map(|iv| match iv {
            Intervention::Set { variable, .. } => Some(*variable),
            _ => None,
        })
        .collect()
}

fn score_table_treatment_col(analysis: &Study, table: &ScoreTable) -> Option<Vec<f64>> {
    let DataInput::Tabular(data) = &analysis.data else {
        return None;
    };
    let values = data.float64_values(table.treatment).ok()?;
    let mut col = Vec::with_capacity(table.n_rows);
    for &idx in table.row_index.iter() {
        col.push(*values.get(idx as usize)?);
    }
    Some(col)
}

/// Whether the query's outcome functional is computed from the frozen score table
/// (exceedance, exceedance grid, quantile) rather than by the estimator itself.
fn query_reads_score_table(query: &CausalQuery) -> bool {
    let functional = match query {
        CausalQuery::AverageEffect(q) => &q.outcome_functional,
        CausalQuery::Response(q) => &q.outcome_functional,
        CausalQuery::ConditionalEffect(q) => &q.inner.outcome_functional,
        _ => return false,
    };
    matches!(
        functional,
        OutcomeFunctional::Exceedance(_)
            | OutcomeFunctional::ExceedanceGrid(_)
            | OutcomeFunctional::Quantile(_)
    )
}

fn overlay_prepared_score_functional(
    query: &CausalQuery,
    table: Option<&ScoreTable>,
    result: &mut StudyResult,
) -> Result<(), CausalError> {
    let Some(table) = table else {
        return Ok(());
    };
    if matches!(query, CausalQuery::AverageEffect(_))
        && result.logical_plan.estimator.as_deref() != Some("aipw")
    {
        return Ok(());
    }
    if !query_reads_score_table(query) {
        return Ok(());
    }
    let functional = match query {
        CausalQuery::AverageEffect(q) => &q.outcome_functional,
        CausalQuery::Response(q) => &q.outcome_functional,
        CausalQuery::ConditionalEffect(q) => &q.inner.outcome_functional,
        _ => return Ok(()),
    };
    let (estimate, diagnostics) = if let Some(tau) = functional.quantile_level() {
        if let CausalQuery::Response(q) = query {
            super::helpers::attach_joint_quantile_from_table(
                result.estimate.clone(),
                table.clone(),
                q,
                tau,
            )?
        } else {
            super::helpers::attach_quantile_from_table(result.estimate.clone(), table.clone(), tau)?
        }
    } else {
        super::helpers::attach_score_functional_grid(result.estimate.clone(), table.clone())?
    };
    if functional.quantile_level().is_some() {
        if let Some(response) = &mut result.response {
            response.estimate = antecedent_core::ResponseIdentification::PointIdentified(
                antecedent_core::ResponseValue::Scalar(estimate.ate),
            );
            response.uncertainty = antecedent_core::ResponseUncertainty::Scalar {
                standard_error: estimate.se_analytic,
                lower: estimate.ate
                    - crate::result::reported_se_interval_z() * estimate.se_analytic,
                upper: estimate.ate
                    + crate::result::reported_se_interval_z() * estimate.se_analytic,
                level: 0.95,
                interpretation: antecedent_core::IntervalInterpretation::Confidence,
                draws: None,
            };
        }
    }
    result.estimate = estimate;
    result.rebind_interval(result.posterior.is_some());
    result.diagnostics.extend(diagnostics);
    Ok(())
}

/// Identification schedule of a sequence plan: `(variable, lag, optional level)` per step.
type IdentificationSchedule = Vec<(antecedent_core::VariableId, i32, Option<f64>)>;

fn sequence_identification_schedule(
    query: &antecedent_core::ResponseQuery,
) -> Result<Option<IdentificationSchedule>, CausalError> {
    match antecedent_estimate::plan_from_response_query(query) {
        Ok(Some(plan)) if plan.mechanism_overlays().is_some() => {
            let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
                message: "sequence schedule requires TemporalResponseSpec".into(),
            })?;
            Ok(Some(plan.identification_schedule(temporal)))
        }
        Ok(_) => Ok(None),
        Err(error) => Err(CausalError::from(error)),
    }
}

/// Requested hard-set dose of a single-step temporal response, when it names one.
pub(crate) fn single_step_dose(
    query: &antecedent_core::ResponseQuery,
) -> Result<Option<f64>, CausalError> {
    match antecedent_estimate::plan_from_response_query(query) {
        Ok(Some(antecedent_estimate::TemporalInterventionPlan::Single { level, .. })) => Ok(level),
        Ok(_) => Ok(None),
        Err(error) => Err(CausalError::from(error)),
    }
}

/// `dose` is the single-step active level (control stays at 0); `None` keeps the unit contrast.
pub(crate) fn identify_temporal_response_horizons(
    graph: &TemporalDag,
    treatment: antecedent_core::VariableId,
    outcome: antecedent_core::VariableId,
    temporal: &TemporalResponseSpec,
    target_population: &TargetPopulation,
    estimator_id: crate::strategy_table::EstimatorId,
    schedule: Option<&[(antecedent_core::VariableId, i32, Option<f64>)]>,
    dose: Option<f64>,
) -> Result<CachedTemporalIdentification, CausalError> {
    use crate::strategy_table::select_estimand;
    if temporal.horizons.is_empty() {
        return Err(CausalError::Compile {
            message: "temporal response requires at least one horizon".into(),
        });
    }
    let origin =
        temporal.treatment_offset().map_err(|e| CausalError::Compile { message: e.to_string() })?;
    let sequential = schedule
        .is_some_and(|nodes| nodes.len() != 1 || (nodes[0].0, nodes[0].1) != (treatment, origin));
    let mut by_horizon = Vec::with_capacity(temporal.horizons.len());
    for &horizon in temporal.horizons.iter() {
        let id_res = if sequential {
            let outcome_at = i32::try_from(horizon.saturating_sub(1)).unwrap_or(i32::MAX);
            TemporalBackdoorIdentifier::new()
                .identify_temporal_schedule(
                    graph,
                    outcome,
                    outcome_at,
                    schedule.expect("sequential schedule"),
                    temporal.max_history_lag,
                    target_population.clone(),
                )
                .map_err(CausalError::from)?
        } else {
            let id_query = TemporalEffectQuery {
                treatment,
                outcome,
                policy: temporal.policy.clone(),
                control: Intervention::set(treatment, Value::f64(0.0)),
                active: Intervention::set(treatment, Value::f64(dose.unwrap_or(1.0))),
                horizon_steps: horizon,
                max_history_lag: temporal.max_history_lag,
                target_population: target_population.clone(),
            };
            TemporalBackdoorIdentifier::new()
                .identify_temporal(graph, &id_query)
                .map_err(CausalError::from)?
        };
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

pub(crate) fn identify_temporal_mediation_horizons(
    graph: &TemporalDag,
    query: &MediationQuery,
    estimator_id: crate::strategy_table::EstimatorId,
) -> Result<CachedTemporalIdentification, CausalError> {
    use crate::strategy_table::select_estimand;
    query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
    if query.horizons.is_empty() {
        return Err(CausalError::Compile {
            message: "temporal mediation requires at least one horizon".into(),
        });
    }
    let ider = TemporalMediationIdentifier {
        allow_natural_controlled_alias: true,
        ..TemporalMediationIdentifier::new()
    };
    let mut by_horizon = Vec::with_capacity(query.horizons.len());
    for &horizon in query.horizons.iter() {
        let (identification, temporal) =
            ider.identify_with_horizon(graph, query, horizon).map_err(CausalError::from)?;
        let estimand = select_estimand(&identification, estimator_id)?;
        by_horizon.push(CachedTemporalHorizonIdentification {
            horizon,
            identification,
            estimand,
            indexer: temporal.indexer,
        });
    }
    Ok(CachedTemporalIdentification { by_horizon: Arc::from(by_horizon) })
}

fn ensure_prepared_supported(analysis: &Study) -> Result<(), CausalError> {
    if analysis.graph_posterior.is_some() {
        // Refuse here what every estimate click would refuse, before the
        // posterior identification cache is built.
        match (&analysis.data, &analysis.query) {
            (DataInput::Tabular(_), CausalQuery::Response(query)) => {
                super::execute::graph_posterior_response_supported(query)?;
            }
            (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::Mediation(query))
                if matches!(analysis.inference, InferenceMode::Frequentist)
                    && query.horizons.len() != 1 =>
            {
                return Err(CausalError::Unsupported {
                    message: "Frequentist DBN-posterior mediation is licensed for one horizon; \
                              multi-horizon grids need their own joint uncertainty contract",
                });
            }
            (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::Response(query)) => {
                super::execute::dbn_posterior_response_supported(query)?;
            }
            _ => {}
        }
        return match (&analysis.data, &analysis.query) {
            (
                DataInput::Tabular(_),
                CausalQuery::AverageEffect(_)
                | CausalQuery::Response(_)
                | CausalQuery::ConditionalEffect(_),
            )
            | (
                DataInput::Temporal(_) | DataInput::Event(_),
                CausalQuery::TemporalEffect(_)
                | CausalQuery::Mediation(_)
                | CausalQuery::Response(_),
            ) => Ok(()),
            _ => Err(crate::support_reason!(
                "data_modality_not_licensed",
                "graph_posterior on the prepared handle is licensed only for tabular \
                 AverageEffect/Response/ConditionalEffect and series TemporalEffect, \
                 TemporalMediationEffect, or temporal Response"
            )),
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
            // An Admg is licensed for a joint response with or without a tiered background
            // (the functional-effect estimator serves it), so no tier condition narrows it.
            if !matches!(
                analysis.graph.class(),
                GraphClass::Dag | GraphClass::Cpdag | GraphClass::Pag | GraphClass::Admg
            ) {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy supports ResponseCurve on a supplied Dag, Cpdag, Pag, \
                              or Admg (or CoDetermined joint cells)",
                });
            }
        }
        (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::Response(q))
            if q.is_temporal() =>
        {
            if !matches!(
                analysis.graph.class(),
                GraphClass::TemporalDag | GraphClass::TemporalCpdag | GraphClass::TemporalPag
            ) {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy supports temporal ResponseCurve on TemporalDag, \
                              TemporalCpdag, or TemporalPag",
                });
            }
        }
        (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::TemporalEffect(_)) => {
            if !matches!(
                analysis.graph.class(),
                GraphClass::TemporalDag | GraphClass::TemporalCpdag | GraphClass::TemporalPag
            ) {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy supports TemporalEffect on TemporalDag, \
                              TemporalCpdag, or TemporalPag",
                });
            }
        }
        (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::Mediation(_)) => {
            if !matches!(
                analysis.graph.class(),
                GraphClass::TemporalDag | GraphClass::TemporalCpdag
            ) {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy supports TemporalMediationEffect on TemporalDag or \
                              TemporalCpdag",
                });
            }
        }
        (DataInput::Tabular(_), CausalQuery::Counterfactual(_)) => {
            if analysis.graph.class() != GraphClass::Dag {
                return Err(CausalError::Unsupported { message: "counterfactual requires Dag" });
            }
        }
        (
            DataInput::Tabular(_),
            CausalQuery::AnomalyAttribution(_) | CausalQuery::ChangeAttribution(_),
        ) => {
            if analysis.graph.class() != GraphClass::Dag {
                return Err(CausalError::Unsupported {
                    message: "AnomalyAttribution and ChangeAttribution require a supplied Dag",
                });
            }
        }
        (DataInput::Tabular(_), CausalQuery::Transport(_)) => {
            if analysis.graph.class() != GraphClass::Admg {
                return Err(CausalError::Unsupported { message: "TransportQuery requires Admg" });
            }
        }
        (DataInput::Tabular(_), CausalQuery::Interference(_)) => {
            if analysis.graph.class() != GraphClass::Dag {
                return Err(CausalError::Unsupported { message: "InterferenceQuery requires Dag" });
            }
        }
        (DataInput::Tabular(_), CausalQuery::Mediation(_)) => {
            if analysis.graph.class() != GraphClass::Dag {
                return Err(CausalError::Unsupported { message: "static mediation requires Dag" });
            }
        }
        (DataInput::Tabular(_), CausalQuery::ConditionalEffect(_)) => {
            if !matches!(
                analysis.graph.class(),
                GraphClass::Dag | GraphClass::Cpdag | GraphClass::Pag
            ) {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy supports ConditionalEffect on a supplied Dag, Cpdag, or Pag",
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
            if !matches!(analysis.graph.class(), GraphClass::Dag | GraphClass::Admg) {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy supports Distribution on a supplied Dag or Admg",
                });
            }
        }
        (DataInput::MultiEnv(_), CausalQuery::TemporalEffect(_)) => {
            if analysis.graph.class() != GraphClass::TemporalDag {
                return Err(crate::unsupported_reason!(
                    "data_modality_not_licensed",
                    "PreparedStudy supports multi-environment TemporalEffect on a TemporalDag"
                ));
            }
        }
        (DataInput::Panel(panel), query) => {
            if !matches!(
                (query, analysis.graph.class()),
                (
                    CausalQuery::TemporalEffect(_) | CausalQuery::Response(_),
                    GraphClass::TemporalDag | GraphClass::TemporalCpdag | GraphClass::TemporalPag
                )
            ) {
                return Err(crate::unsupported_reason!(
                    "data_modality_not_licensed",
                    "PreparedStudy supports panel Pulse/Sustained and temporal response on \
                     TemporalDag, TemporalCpdag, or TemporalPag"
                ));
            }
            super::builder::refuse_unlicensed_panel_route(
                query,
                analysis.graph.class(),
                &analysis.inference,
                panel,
                analysis.split.as_ref(),
            )?;
        }
        _ => {
            return Err(crate::unsupported_reason!(
                "data_modality_not_licensed",
                "PreparedStudy supports AverageEffect, ResponseCurve, ConditionalEffect, \
                 PathSpecific, Distribution, temporal ResponseCurve, TemporalEffect (Pulse / \
                 single-step Sustained), TemporalMediationEffect, panel Pulse/Sustained, \
                 Counterfactual, AnomalyAttribution, ChangeAttribution, TransportQuery, or \
                 InterferenceQuery"
            ));
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
    use super::{is_supplied_static_graph, query_reads_score_table};
    use crate::accepted::GraphClass;
    use antecedent_core::{AverageEffectQuery, CausalQuery, OutcomeFunctional, VariableId};

    /// A mean click keeps the estimator's own value, so refitting the cross-fit score table
    /// for it would be discarded work; only exceedance, grid and quantile clicks read it.
    #[test]
    fn only_non_mean_functionals_read_the_frozen_score_table() {
        let base = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        assert!(!query_reads_score_table(&CausalQuery::AverageEffect(base.clone())));
        for functional in [
            OutcomeFunctional::exceedance(0.5),
            OutcomeFunctional::exceedance_grid(vec![0.0, 1.0]),
            OutcomeFunctional::quantile(0.5),
        ] {
            let query = base.clone().with_outcome_functional(functional);
            assert!(query_reads_score_table(&CausalQuery::AverageEffect(query)));
        }
    }

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

#[cfg(test)]
mod refresh_tests {
    use std::sync::Arc;

    use antecedent_core::{
        CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint, SmallRoleSet,
        TemporalEffectQuery, TemporalPolicy, ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TableView, TimeIndex,
        TimeSeriesData, ValidityBitmap,
    };
    use antecedent_graph::{TemporalDag, ensure_lagged};

    use super::super::builder::{DataInput, RefuteSuite};
    use crate::analysis::execute::Study;

    #[allow(clippy::cast_precision_loss)]
    fn xy_series(n: usize) -> TimeSeriesData {
        let mut b = CausalSchemaBuilder::new();
        for (name, hint) in [("x", RoleHint::TreatmentCandidate), ("y", RoleHint::OutcomeCandidate)]
        {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(hint),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        for t in 1..n {
            x[t] = ((t as f64) * 0.07).sin();
            y[t] = 0.8 * x[t - 1];
        }
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(x),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap()
    }

    fn retained_rows(prepared: &super::PreparedStudy) -> usize {
        match &prepared.analysis.data {
            DataInput::Temporal(data) => data.row_count(),
            _ => panic!("series handle must retain series data"),
        }
    }

    #[test]
    fn failed_series_refresh_leaves_handle_unchanged() {
        let mut graph = TemporalDag::empty();
        let x1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(x1, y0).unwrap();
        let query =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_policy(TemporalPolicy::pulse(-1))
                .with_horizon_steps(1)
                .with_max_history_lag(Some(1));
        let ctx = ExecutionContext::for_tests(2);
        let mut prepared = Study::series(xy_series(160))
            .graph(graph)
            .temporal_query(query)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .prepare(&ctx)
            .unwrap();
        assert_eq!(retained_rows(&prepared), 160);
        let original = xy_series(160);
        let before = prepared.contract().unwrap().identities;

        // Same schema and regularity, so the compatibility gate passes and the
        // failure comes from estimation itself.
        assert!(prepared.refresh_series(xy_series(2), &ctx).is_err());
        assert_eq!(retained_rows(&prepared), 160, "failed refresh must not replace data");
        assert_eq!(before, prepared.contract().unwrap().identities);
        let recovered = prepared.estimate_series(&original, &ctx).unwrap();
        assert!(recovered.effect().is_finite(), "old-data estimate must survive a failed refresh");
        assert_eq!(before, prepared.contract().unwrap().identities);

        prepared.refresh_series(xy_series(140), &ctx).unwrap();
        assert_eq!(retained_rows(&prepared), 140);
        let after = prepared.contract().unwrap().identities;
        assert_eq!(before.program, after.program);
        assert_ne!(before.data_snapshot, after.data_snapshot);
    }
}

#[cfg(test)]
#[path = "dbn_mediation_cache_tests.rs"]
mod dbn_mediation_cache_tests;
