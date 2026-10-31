//! DBN graph-posterior identification caches: contrast, response and mediation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ExecutionContext, MediationQuery, ResponseQuery, TemporalEffectQuery};
use antecedent_data::TemporalIndexer;
use antecedent_discovery::GraphPosterior;
use antecedent_expr::IdentifiedEstimand;
use antecedent_graph::TemporalDag;
use antecedent_identify::{IdentificationResult, TemporalBackdoorIdentifier};
use antecedent_prob::{GraphIdentFlag, WeightedGraphSamples};

use super::{dbn_envelope_key, map_posterior_graphs};
use crate::analysis::prepared::{
    CachedTemporalIdentification, TemporalResponseIdentificationRequest,
    identify_temporal_response_horizons, single_step_dose,
};
use crate::error::CausalError;

/// Admit a DBN template conversion before scanning lag/variable combinations.
pub(crate) fn admitted_temporal_dag_atom(
    posterior: &GraphPosterior,
    index: usize,
    variables: &[antecedent_core::VariableId],
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<Option<(TemporalDag, antecedent_core::execution::AdmissionLease)>, CausalError> {
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: "temporal atom" });
    }
    let masks = posterior.lag_masks().ok_or_else(|| CausalError::Compile {
        message: "DBN posterior missing lag masks".into(),
    })?;
    let Some((cmask, lmask)) = posterior.atoms().get(index).zip(masks.get(index)) else {
        return Ok(None);
    };
    let lag = posterior
        .max_lag()
        .ok_or_else(|| CausalError::Compile { message: "DBN posterior missing max_lag".into() })?;
    let n = posterior.n_vars() as u64;
    let scans = n
        .checked_mul(n)
        .and_then(|n| n.checked_mul(u64::from(lag) + 1))
        .ok_or_else(|| CausalError::Resource { message: "DBN template scan overflow".into() })?;
    let words = (cmask.words().len() as u64)
        .checked_add(lmask.words().len() as u64)
        .ok_or_else(|| CausalError::Resource { message: "DBN mask size overflow".into() })?;
    budget.charge_work(words).map_err(|e| CausalError::Resource { message: e.to_string() })?;
    let mut edges = 0u64;
    for word in cmask.words().iter().chain(lmask.words()) {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "temporal atom" });
        }
        edges = edges
            .checked_add(u64::from(word.count_ones()))
            .ok_or_else(|| CausalError::Resource { message: "DBN edge count overflow".into() })?;
    }
    let nodes_by_edges = edges
        .checked_mul(2)
        .and_then(|v| v.checked_add(n))
        .ok_or_else(|| CausalError::Resource { message: "DBN node count overflow".into() })?;
    let nodes_by_lag = n
        .checked_mul(u64::from(lag) + 1)
        .ok_or_else(|| CausalError::Resource { message: "DBN node count overflow".into() })?;
    let nodes = nodes_by_edges.min(nodes_by_lag);
    if nodes > budget.limits().max_graph_nodes || edges > budget.limits().max_graph_edges {
        return Err(CausalError::Resource { message: "DBN template graph limit".into() });
    }
    let work = scans
        .checked_mul(n.max(1))
        .and_then(|w| {
            edges
                .checked_mul(nodes.checked_add(edges)?.checked_add(1)?)
                .and_then(|e| w.checked_add(e))
        })
        .ok_or_else(|| CausalError::Resource { message: "DBN template work overflow".into() })?;
    budget.charge_work(work).map_err(|e| CausalError::Resource { message: e.to_string() })?;
    let bytes = nodes
        .checked_mul(512)
        .and_then(|v| edges.checked_mul(256).and_then(|e| v.checked_add(e)))
        .and_then(|v| v.checked_add(4096))
        .ok_or_else(|| CausalError::Resource { message: "DBN template bytes overflow".into() })?;
    if bytes > budget.limits().max_bytes {
        return Err(CausalError::Resource { message: "DBN template byte limit".into() });
    }
    let lease = ctx.acquire_admission(0, bytes).map_err(|e| match e {
        antecedent_core::execution::AdmissionError::Cancelled => {
            CausalError::Cancelled { stage: "temporal atom" }
        }
        other => CausalError::Resource { message: other.to_string() },
    })?;
    let graph = antecedent_discovery::dbn_posterior::temporal_dag_from_dbn_atoms_cancellable(
        cmask,
        lmask,
        posterior.n_vars(),
        lag,
        variables,
        || ctx.cancellation.is_cancelled(),
    );
    match graph {
        Ok(Some(graph)) => Ok(Some((graph, lease))),
        Ok(None) => Err(CausalError::Cancelled { stage: "temporal atom" }),
        Err(_) => Ok(None),
    }
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
    /// `temporal_dag_from_dbn_atoms` rejected the atom.
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
    /// Horizon projections already built from this cache, so repeated
    /// projections of one horizon copy shared handles instead of every atom's
    /// identification products.
    projections: HorizonProjections,
}

/// Memo of [`CachedDbnPosteriorIdentification::mediation_horizon`] results.
///
/// A clone starts empty: the memo belongs to the cache value that built it, so
/// a clone whose atoms are later replaced can never serve a stale projection.
#[derive(Debug, Default)]
struct HorizonProjections(
    std::sync::Mutex<std::collections::HashMap<u32, CachedDbnPosteriorIdentification>>,
);

impl Clone for HorizonProjections {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl CachedDbnPosteriorIdentification {
    /// Project the cached union of eligible mediation atoms onto one horizon.
    /// An atom's failure at another horizon never changes this horizon's mass.
    ///
    /// The first projection of a horizon builds it; later calls clone the
    /// memoised projection, whose atoms and weights are shared handles.
    pub fn mediation_horizon(&self, horizon: u32) -> Result<Self, CausalError> {
        fn memo(
            projections: &HorizonProjections,
        ) -> std::sync::MutexGuard<
            '_,
            std::collections::HashMap<u32, CachedDbnPosteriorIdentification>,
        > {
            projections.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
        }
        if let Some(found) = memo(&self.projections).get(&horizon) {
            return Ok(found.clone());
        }
        let projected = self.project_mediation_horizon(horizon)?;
        memo(&self.projections).entry(horizon).or_insert_with(|| projected.clone());
        Ok(projected)
    }

    fn project_mediation_horizon(&self, horizon: u32) -> Result<Self, CausalError> {
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
                let entry = atom.horizons.as_ref()?.get(horizon)?;
                Some(CachedDbnPosteriorAtomIdentification {
                    key: atom.key,
                    estimand: entry.estimand.clone(),
                    identification: entry.identification.clone(),
                    indexer: entry.indexer.clone(),
                    horizons: Some(CachedTemporalIdentification {
                        by_horizon: Arc::from([entry.clone()]),
                    }),
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
        )?;
        let horizon_demotions = Arc::from([(horizon, identify_demotion.clone())]);
        Ok(Self {
            graphs,
            atoms: atoms.into(),
            identify_demotion,
            horizon_demotions,
            projections: HorizonProjections::default(),
        })
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

    let _lag_masks = posterior.lag_masks().ok_or_else(|| CausalError::Compile {
        message: "DBN posterior missing per-atom lag masks".into(),
    })?;
    let _max_lag = posterior
        .max_lag()
        .ok_or_else(|| CausalError::Compile { message: "DBN posterior missing max_lag".into() })?;
    crate::analysis::execute::report_identify_compute(ctx);
    let budget = antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
    let mapped = map_posterior_graphs(posterior, ctx, |i, inner| {
        let mut budget = budget.clone();
        // A DBN atom is the pair (contemporaneous mask, lag mask), but the
        // public GraphPosterior constructor keys atoms by contemporaneous mask
        // alone. Use posterior position as a collision-free execution key so
        // lag-distinct atoms keep distinct fits, flags, and weights inside the
        // effect envelope. This key is internal and does not alter the public
        // GraphPosterior representation.
        let key = dbn_envelope_key(i)?;
        let Some((graph, _graph_lease)) =
            admitted_temporal_dag_atom(posterior, i, variables, &mut budget, inner)?
        else {
            return Ok(DbnAtomOutcome::InvalidGraph);
        };
        let temporal = match TemporalBackdoorIdentifier::new()
            .with_temporal_context(inner, &budget)
            .identify_temporal(&graph, query)
        {
            Ok(value) => value,
            Err(antecedent_identify::IdentificationError::Cancelled) => {
                return Err(CausalError::Cancelled { stage: "temporal identification" });
            }
            Err(error @ antecedent_identify::IdentificationError::Budget { .. }) => {
                return Err(CausalError::from(error));
            }
            Err(_) => return Ok(DbnAtomOutcome::IdentifyFailed),
        };
        let identification = temporal.result;
        if !crate::analysis::execute::identification_status_ok_for_case(identification.status) {
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
    let mut weights = Vec::with_capacity(posterior.n_graphs());
    let mut flags = Vec::with_capacity(posterior.n_graphs());
    let mut keys = Vec::with_capacity(posterior.n_graphs());
    let mut atoms = Vec::new();
    let mut identify_demotion = DbnIdentifyDemotion::default();

    for (i, outcome) in mapped.into_iter().enumerate() {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_IDENTIFY });
        }
        if let Some(progress) = &ctx.progress {
            progress.report(i as f64 / posterior.n_graphs().max(1) as f64, "envelope.identify");
        }
        let key = dbn_envelope_key(i)?;
        keys.push(key);
        weights.push(posterior.weights()[i]);
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
        return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_IDENTIFY });
    }

    let graphs = WeightedGraphSamples::new(weights, flags, keys)?;
    Ok(CachedDbnPosteriorIdentification {
        graphs,
        atoms: Arc::from(atoms),
        identify_demotion,
        horizon_demotions: Arc::from([]),
        projections: HorizonProjections::default(),
    })
}

/// Identify every DBN atom for a temporal [`antecedent_core::CausalQuery::Response`].
///
/// Each atom is reconstructed as a [`TemporalDag`] and identified at every
/// requested horizon. An atom that fails any horizon stays unidentified;
/// unidentified mass is retained. Atoms are DAGs: there is no completion
/// enumeration.
#[allow(
    clippy::too_many_lines,
    reason = "validates the DBN response posterior, identifies each atom's horizons under cancellation and records the demotion reasons while walking the samples once"
)]
pub(crate) fn build_dbn_posterior_response_identification_cache(
    posterior: &GraphPosterior,
    variables: &[antecedent_core::VariableId],
    query: &ResponseQuery,
    estimator_id: crate::strategy_table::EstimatorId,
    ctx: &ExecutionContext,
) -> Result<CachedDbnPosteriorIdentification, CausalError> {
    crate::analysis::execute::dbn_posterior_response_supported(query)?;
    let temporal = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
        message: "DBN-posterior response requires TemporalResponseSpec".into(),
    })?;
    let (treatment, outcome) =
        crate::analysis::proof_support::require_primary_pair(&query.functional)?;
    let _lag_masks = posterior.lag_masks().ok_or_else(|| CausalError::Compile {
        message: "DBN posterior missing per-atom lag masks".into(),
    })?;
    let _max_lag = posterior
        .max_lag()
        .ok_or_else(|| CausalError::Compile { message: "DBN posterior missing max_lag".into() })?;
    crate::analysis::execute::report_identify_compute(ctx);
    let budget = antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
    let dose = single_step_dose(query)?;
    let mapped = map_posterior_graphs(posterior, ctx, |i, inner| {
        let mut budget = budget.clone();
        let key = dbn_envelope_key(i)?;
        let Some((graph, _graph_lease)) =
            admitted_temporal_dag_atom(posterior, i, variables, &mut budget, inner)?
        else {
            return Ok(DbnAtomOutcome::InvalidGraph);
        };
        let horizons = match identify_temporal_response_horizons(
            &graph,
            estimator_id,
            &TemporalResponseIdentificationRequest {
                treatment,
                outcome,
                temporal,
                target_population: &query.target_population,
                schedule: None,
                dose,
            },
            &mut budget,
            inner,
        ) {
            Ok(horizons) => horizons,
            Err(error @ (CausalError::Resource { .. } | CausalError::Cancelled { .. })) => {
                return Err(error);
            }
            Err(_) => return Ok(DbnAtomOutcome::IdentifyFailed),
        };
        if horizons.by_horizon.len() != temporal.horizons.len()
            || horizons.by_horizon.iter().any(|entry| {
                !crate::analysis::execute::identification_status_ok_for_case(
                    entry.identification.status,
                ) || entry.identification.estimands.is_empty()
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
    let mut weights = Vec::with_capacity(posterior.n_graphs());
    let mut flags = Vec::with_capacity(posterior.n_graphs());
    let mut keys = Vec::with_capacity(posterior.n_graphs());
    let mut atoms = Vec::new();
    let mut identify_demotion = DbnIdentifyDemotion::default();

    for (i, outcome) in mapped.into_iter().enumerate() {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_IDENTIFY });
        }
        if let Some(progress) = &ctx.progress {
            progress.report(i as f64 / posterior.n_graphs().max(1) as f64, "envelope.identify");
        }
        let key = dbn_envelope_key(i)?;
        keys.push(key);
        weights.push(posterior.weights()[i]);
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
        return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_IDENTIFY });
    }

    let graphs = WeightedGraphSamples::new(weights, flags, keys)?;
    Ok(CachedDbnPosteriorIdentification {
        graphs,
        atoms: Arc::from(atoms),
        identify_demotion,
        horizon_demotions: Arc::from([]),
        projections: HorizonProjections::default(),
    })
}

/// Identify every DBN atom for [`antecedent_core::CausalQuery::Mediation`] (`TemporalMediationEffect`).
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
    let budget = antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
    build_dbn_mediation_cache_with_identifier_within(
        posterior,
        variables,
        query,
        ctx,
        budget.clone(),
        |_, _, graph, horizon_query| {
            crate::analysis::prepared::identify_temporal_mediation_horizons_within(
                graph,
                horizon_query,
                crate::strategy_table::EstimatorId::BayesianTemporalMediation,
                &mut budget.clone(),
                ctx,
            )
        },
    )
}

#[cfg(test)]
pub(crate) fn build_dbn_mediation_cache_with_identifier(
    posterior: &GraphPosterior,
    variables: &[antecedent_core::VariableId],
    query: &MediationQuery,
    ctx: &ExecutionContext,
    identify: impl FnMut(
        usize,
        u32,
        &TemporalDag,
        &MediationQuery,
    ) -> Result<CachedTemporalIdentification, CausalError>,
) -> Result<CachedDbnPosteriorIdentification, CausalError> {
    let budget = antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
    build_dbn_mediation_cache_with_identifier_within(
        posterior, variables, query, ctx, budget, identify,
    )
}

fn build_dbn_mediation_cache_with_identifier_within(
    posterior: &GraphPosterior,
    variables: &[antecedent_core::VariableId],
    query: &MediationQuery,
    ctx: &ExecutionContext,
    mut budget: antecedent_core::TemporalBudget,
    mut identify: impl FnMut(
        usize,
        u32,
        &TemporalDag,
        &MediationQuery,
    ) -> Result<CachedTemporalIdentification, CausalError>,
) -> Result<CachedDbnPosteriorIdentification, CausalError> {
    let _lag_masks = posterior.lag_masks().ok_or_else(|| CausalError::Compile {
        message: "DBN posterior missing per-atom lag masks".into(),
    })?;
    let _max_lag = posterior
        .max_lag()
        .ok_or_else(|| CausalError::Compile { message: "DBN posterior missing max_lag".into() })?;
    crate::analysis::execute::report_identify_compute(ctx);
    let mut weights = Vec::with_capacity(posterior.n_graphs());
    let mut flags = Vec::with_capacity(posterior.n_graphs());
    let mut keys = Vec::with_capacity(posterior.n_graphs());
    let mut atoms = Vec::new();
    let _validation = antecedent_core::admit_temporal_intervention_validation(
        [&query.control, &query.active],
        &mut budget,
        ctx,
    )
    .map_err(|e| {
        CausalError::from(antecedent_estimate::EstimationError::TemporalMaterialization(e))
    })?;
    budget
        .charge_work((query.mediators.len() as u64).saturating_add(query.horizons.len() as u64))
        .map_err(|e| {
            CausalError::from(antecedent_estimate::EstimationError::TemporalMaterialization(e))
        })?;
    query.validate()?;
    let mut horizon_demotions: Vec<_> =
        query.horizons.iter().map(|h| (*h, DbnIdentifyDemotion::default())).collect();

    for i in 0..posterior.n_graphs() {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_IDENTIFY });
        }
        if let Some(progress) = &ctx.progress {
            progress.report(i as f64 / posterior.n_graphs().max(1) as f64, "envelope.identify");
        }
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_IDENTIFY });
        }
        let key = dbn_envelope_key(i)?;
        keys.push(key);
        weights.push(posterior.weights()[i]);
        let Some((graph, _graph_lease)) =
            admitted_temporal_dag_atom(posterior, i, variables, &mut budget, ctx)?
        else {
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
            let horizons = match identify(i, *horizon, &graph, &horizon_query) {
                Ok(horizons) => horizons,
                Err(error @ (CausalError::Resource { .. } | CausalError::Cancelled { .. })) => {
                    return Err(error);
                }
                Err(_) => {
                    counts.identify_failed += 1;
                    continue;
                }
            };
            let Some(entry) = horizons.by_horizon.first() else {
                counts.no_estimand += 1;
                continue;
            };
            if !crate::analysis::execute::identification_status_ok_for_case(
                entry.identification.status,
            ) || entry.identification.estimands.is_empty()
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
        return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_IDENTIFY });
    }

    let graphs = WeightedGraphSamples::new(weights, flags, keys)?;
    Ok(CachedDbnPosteriorIdentification {
        graphs,
        atoms: atoms.into(),
        identify_demotion: DbnIdentifyDemotion::default(),
        horizon_demotions: horizon_demotions.into(),
        projections: HorizonProjections::default(),
    })
}

fn dbn_temporal_effect_estimator(
    query: &TemporalEffectQuery,
) -> crate::strategy_table::EstimatorId {
    crate::strategy_table::EstimatorId::temporal_effect_procedure(query, false)
}
