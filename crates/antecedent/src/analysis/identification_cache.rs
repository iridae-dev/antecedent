//! Prepare-time identification caches for graph-posterior and DBN-posterior
//! effect, response, and mediation queries.
//!
//! Extracted verbatim from `analysis::prepared`; behaviour, ordering, and
//! numerics are unchanged.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, CausalQuery, ExecutionContext, Intervention, MediationQuery, ResponseQuery,
    TemporalEffectQuery, Value,
};
use antecedent_data::TemporalIndexer;
use antecedent_discovery::{
    GraphPosterior, dag_from_adjacency_mask, temporal_cpdag_from_dbn_masks,
    temporal_dag_from_dbn_masks, temporal_pag_from_dbn_masks,
};
use antecedent_expr::IdentifiedEstimand;
use antecedent_graph::TemporalDag;
use antecedent_identify::{
    IdentificationEnvelope, IdentificationResult, IdentificationStatus, TemporalBackdoorIdentifier,
};
use antecedent_prob::{GraphIdentFlag, WeightedGraphSamples};

use crate::error::CausalError;

use super::prepared::{
    CachedTemporalIdentification, identify_temporal_mediation_horizons,
    identify_temporal_response_horizons, single_step_dose,
};

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

pub(crate) fn build_dbn_mediation_cache_with_identifier(
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
    crate::strategy_table::EstimatorId::temporal_effect_procedure(query, false)
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
