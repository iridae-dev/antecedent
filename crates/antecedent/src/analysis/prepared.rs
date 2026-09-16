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
    OutcomeFunctional, TargetPopulation, TemporalEffectQuery, TemporalResponseSpec, Value,
};
use antecedent_data::{PanelData, TableView, TabularData, TemporalIndexer, TimeSeriesData};
use antecedent_discovery::{GraphPosterior, dag_from_adjacency_mask, temporal_dag_from_dbn_masks};
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

/// Prepare-time identification for every atom in a static graph posterior.
#[derive(Clone, Debug)]
pub(crate) struct CachedGraphPosteriorIdentification {
    /// Frozen weights, graph keys, and identified/unidentified flags.
    pub graphs: WeightedGraphSamples,
    /// Identified atoms, one per distinct graph key, in order of first
    /// appearance. Weight an atom by the combined identified mass of its key in
    /// [`Self::graphs`] (`identified_weight_for_key`), which keeps one entry per
    /// posterior sample. Unidentified atoms remain in [`Self::graphs`] with
    /// [`GraphIdentFlag::Unidentified`].
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
    /// Per-horizon `I(h)` for a mediation atom. Contrast atoms leave this empty
    /// and use [`Self::identification`] / [`Self::indexer`] for the query horizon.
    /// A union of these sets across atoms is not a shared adjustment set.
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
    let mut atom_masks: HashMap<u64, u64> = HashMap::new();

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
                    let Ok(estimand) =
                        select_estimand(&identification, EstimatorId::LinearAdjustmentAte).or_else(
                            |_| select_estimand(&identification, EstimatorId::BayesianGcomp),
                        )
                    else {
                        return Ok(None);
                    };
                    Ok(Some((estimand, identification)))
                })()?;
            by_mask.insert(mask, value.clone());
            value
        };
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
    use crate::strategy_table::select_estimand;

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
        let estimator = dbn_temporal_effect_estimator(query);
        let Ok(estimand) = select_estimand(&identification, estimator) else {
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
            horizons: None,
        });
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
pub struct PreparedStudy {
    /// Frozen analysis config (data slot replaced on each estimate). Read
    /// through [`Self::study`]; mutate only through [`Self::study_mut`], which
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
    #[must_use]
    pub fn temporal_identification(&self) -> Option<&CachedTemporalIdentification> {
        self.analysis.temporal_identification_cache.as_deref()
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
        let table = self.score_table.as_ref().ok_or(CausalError::Unsupported {
            message: "retarget requires a prepared score table on AverageEffect or \
                      discrete joint InterventionResponse",
        })?;
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
        let n_thresholds = {
            let mut t: Vec<f64> = table.columns.iter().filter_map(|c| c.threshold).collect();
            t.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            t.dedup_by(|a, b| *a == *b);
            t.len()
        };
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
                let antecedent_core::ResponseFunctional::InterventionResponse {
                    interventions, ..
                } = &q.functional
                else {
                    return Err(CausalError::Unsupported {
                        message: "retarget requires joint Set response",
                    });
                };
                let mut arm = 0u32;
                for (j, iv) in interventions.iter().enumerate() {
                    let Intervention::Set { value, .. } = iv else {
                        return Err(CausalError::Unsupported {
                            message: "retarget requires Set interventions",
                        });
                    };
                    let v = value.as_f64().ok_or(CausalError::Unsupported {
                        message: "retarget requires numeric binary levels",
                    })?;
                    if (v != 0.0 && v != 1.0) || j >= 3 {
                        return Err(CausalError::Unsupported {
                            message: "retarget requires at most three binary coordinates",
                        });
                    }
                    arm |= u32::from(v == 1.0) << j;
                }
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
            "retarget averages the prepared cross-fitted φ table; it is not a residualized full-sample AIPW refit",
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
                    lower: result.estimate.ate - 1.96 * result.estimate.se_analytic,
                    upper: result.estimate.ate + 1.96 * result.estimate.se_analytic,
                    level: 0.95,
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
        click_analysis.shared_batch_design = shared;
        let mut result = click_analysis.execute_tabular(data, &self.plan, ctx)?;
        let click_scores = click_analysis.prepare_score_table(ctx)?;
        overlay_prepared_score_functional(
            &self.analysis.query,
            click_scores.as_ref(),
            &mut result,
        )?;
        Ok(result)
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
        let mut refreshed = self.analysis.clone();
        refreshed.shared_batch_design = refreshed
            .shared_batch_design
            .as_ref()
            .map(|s| s.rebind(&data).map(Arc::new))
            .transpose()?;
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

fn panel_regularity(panel: &PanelData) -> Result<antecedent_data::SamplingRegularity, CausalError> {
    let first = &panel
        .unit(0)
        .map_err(|e| CausalError::Compile { message: e.to_string() })?
        .series
        .time_index()
        .regularity;
    if panel.units().iter().any(|unit| &unit.series.time_index().regularity != first) {
        return Err(CausalError::Compile {
            message: "PreparedStudy requires every panel unit to share one time-index \
                      regularity; align the units before preparing"
                .into(),
        });
    }
    Ok(first.clone())
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
    ///   on a supplied [`GraphClass::Dag`], [`GraphClass::Cpdag`], or
    ///   [`GraphClass::Pag`] (joint Response also on a CoDetermined tier closure)
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
            DataInput::Panel(panel) => {
                (panel.schema().clone(), PreparedModality::Panel, Some(panel_regularity(panel)?))
            }
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
                let (treatment, outcome) =
                    query.functional.primary_pair().ok_or_else(|| CausalError::Compile {
                        message: "response query has no treatment/outcome pair".into(),
                    })?;
                analysis.graph_posterior_identification_cache =
                    Some(Arc::new(build_graph_posterior_identification_cache(
                        posterior,
                        &AverageEffectQuery::binary_ate(treatment, outcome),
                        ctx,
                    )?));
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
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::Mediation(query),
                Some(posterior),
            ) => {
                let variables: Vec<_> =
                    data.schema().variables().iter().map(|variable| variable.id).collect();
                analysis.dbn_posterior_identification_cache =
                    Some(Arc::new(build_dbn_posterior_mediation_identification_cache(
                        posterior, &variables, query, ctx,
                    )?));
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
                        TemporalEffect or TemporalMediationEffect",
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
            analysis,
            program_cache: std::sync::OnceLock::new(),
            plan,
            schema,
            modality,
            time_regularity,
            score_table,
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
        if matches!(estimator_id, EstimatorId::RdSharp) {
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
                let estimand = select_estimand(&identification, estimator_id)?;
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
                )?))
            }
            CausalQuery::TemporalEffect(query) => {
                let id_res = TemporalBackdoorIdentifier::new()
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
        _ctx: &ExecutionContext,
    ) -> Result<Option<ScoreTable>, CausalError> {
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
                    return Err(CausalError::Unsupported {
                        message: "prepared AIPW scores require iid inference without propensity trimming",
                    });
                }
                let estimand = cache.estimand.clone();
                let mut problem = est.prepare(data, &estimand, query)?;
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
                let est = CellSaturatedAipw::new();
                let continuous = self.continuous_cell.as_ref().map(|(variable, grid)| {
                    antecedent_estimate::ContinuousCellSpec { variable: *variable, grid }
                });
                let (fold_ids, design) = match self.shared_batch_design.as_ref() {
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
                        let folds = if row_index.is_empty() {
                            None
                        } else {
                            Some(shared.folds_for(&row_index)?)
                        };
                        let design = if row_index.is_empty() {
                            None
                        } else {
                            shared.design_for(&cache.estimand.adjustment_set, &row_index)?
                        };
                        (folds, design)
                    }
                    None => (None, None),
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
    let functional = match query {
        CausalQuery::AverageEffect(q) => &q.outcome_functional,
        CausalQuery::Response(q) => &q.outcome_functional,
        CausalQuery::ConditionalEffect(q) => &q.inner.outcome_functional,
        _ => return Ok(()),
    };
    if !matches!(
        functional,
        OutcomeFunctional::Exceedance(_)
            | OutcomeFunctional::ExceedanceGrid(_)
            | OutcomeFunctional::Quantile(_)
    ) {
        return Ok(());
    }
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
            };
        }
    }
    result.estimate = estimate;
    result.rebind_interval(result.posterior.is_some());
    result.diagnostics.extend(diagnostics);
    Ok(())
}

fn sequence_identification_schedule(
    query: &antecedent_core::ResponseQuery,
) -> Result<Option<Vec<(antecedent_core::VariableId, i32)>>, CausalError> {
    match antecedent_estimate::plan_from_response_query(query) {
        Ok(Some(plan)) => Ok(plan
            .mechanism_overlays()
            .map(|overlays| overlays.iter().map(|o| (o.node.variable, o.node.offset)).collect())),
        Ok(_) => Ok(None),
        Err(error) => Err(CausalError::from(error)),
    }
}

pub(crate) fn identify_temporal_response_horizons(
    graph: &TemporalDag,
    treatment: antecedent_core::VariableId,
    outcome: antecedent_core::VariableId,
    temporal: &TemporalResponseSpec,
    target_population: &TargetPopulation,
    estimator_id: crate::strategy_table::EstimatorId,
    schedule: Option<&[(antecedent_core::VariableId, i32)]>,
) -> Result<CachedTemporalIdentification, CausalError> {
    use crate::strategy_table::select_estimand;
    if temporal.horizons.is_empty() {
        return Err(CausalError::Compile {
            message: "temporal response requires at least one horizon".into(),
        });
    }
    let origin =
        temporal.treatment_offset().map_err(|e| CausalError::Compile { message: e.to_string() })?;
    let sequential =
        schedule.is_some_and(|nodes| nodes.len() != 1 || nodes[0] != (treatment, origin));
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
                active: Intervention::set(treatment, Value::f64(1.0)),
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
                CausalQuery::TemporalEffect(_) | CausalQuery::Mediation(_),
            ) => Ok(()),
            _ => Err(CausalError::Support {
                id: crate::support::SupportRefusal::Refused,
                message: "graph_posterior on the prepared handle is licensed only for \
                    tabular AverageEffect/Response/ConditionalEffect and series \
                    TemporalEffect or TemporalMediationEffect",
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
            let codetermined = analysis
                .tiered
                .as_ref()
                .is_some_and(|b| b.within_tier == antecedent_graph::WithinTier::CoDetermined);
            if !(matches!(
                analysis.graph.class(),
                GraphClass::Dag | GraphClass::Cpdag | GraphClass::Pag
            ) || analysis.graph.class() == GraphClass::Admg && codetermined)
            {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy supports ResponseCurve on a supplied Dag, Cpdag, or Pag \
                              (or CoDetermined joint cells)",
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
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy supports multi-env TemporalEffect on TemporalDag",
                });
            }
        }
        (DataInput::Panel(_), query) => {
            if !matches!(
                (query, analysis.graph.class()),
                (
                    CausalQuery::TemporalEffect(_) | CausalQuery::Response(_),
                    GraphClass::TemporalDag | GraphClass::TemporalCpdag | GraphClass::TemporalPag
                )
            ) {
                return Err(CausalError::Unsupported {
                    message: "PreparedStudy supports panel Pulse/Sustained and temporal \
                              response on TemporalDag, TemporalCpdag, or TemporalPag",
                });
            }
            super::builder::refuse_unlicensed_panel_route(
                query,
                analysis.graph.class(),
                &analysis.inference,
            )?;
        }
        _ => {
            return Err(CausalError::Unsupported {
                message: "PreparedStudy currently supports AverageEffect, ResponseCurve, \
                    ConditionalEffect, PathSpecific, Distribution, temporal ResponseCurve, \
                    TemporalEffect (Pulse / single-step Sustained), TemporalMediationEffect, \
                    panel Pulse/Sustained, Counterfactual, AnomalyAttribution, \
                    ChangeAttribution, TransportQuery, or InterferenceQuery",
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
