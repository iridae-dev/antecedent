//! Prepare-time identification caches for graph-posterior and DBN-posterior
//! effect, response, and mediation queries.
//!
//! One file per cache family: static/ADMG/class graph posteriors
//! ([`graph_posterior`]), DBN posteriors ([`dbn`]) and temporal CPDAG/PAG
//! class posteriors ([`temporal_class`]). The helpers every family shares live
//! here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod dbn;
mod exposure_posterior;
mod graph_posterior;
mod temporal_class;

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_discovery::{AdjacencyMask, GraphPosterior};
use antecedent_expr::IdentifiedEstimand;
use antecedent_identify::IdentificationResult;
use antecedent_prob::{GraphIdentFlag, WeightedGraphSamples};

use crate::error::CausalError;

#[cfg(test)]
pub(crate) use self::dbn::build_dbn_mediation_cache_with_identifier;
pub(crate) use self::dbn::{
    CachedDbnPosteriorAtomIdentification, CachedDbnPosteriorIdentification, DbnIdentifyDemotion,
    build_dbn_posterior_identification_cache, build_dbn_posterior_mediation_identification_cache,
    build_dbn_posterior_response_identification_cache, temporal_dag_from_dbn_atom,
};
pub(crate) use self::exposure_posterior::{
    TRANSPORT_CLASS_ATOM_CODE, build_interference_graph_posterior_identification_cache,
    build_transport_graph_posterior_identification_cache,
};
#[cfg(test)]
pub(crate) use self::graph_posterior::CachedClassPosteriorCase;
pub(crate) use self::graph_posterior::{
    CachedClassPosteriorAtomIdentification, CachedGraphPosteriorAtomIdentification,
    CachedGraphPosteriorIdentification, build_admg_graph_posterior_response_identification_cache,
    build_graph_posterior_identification_cache,
    build_temporal_admg_posterior_identification_cache,
};
pub(crate) use self::temporal_class::{
    CachedTemporalClassPosteriorAtomIdentification, CachedTemporalClassPosteriorIdentification,
    build_temporal_class_posterior_identification_cache,
};

/// Identify unique adjacency masks under `ctx.parallelism`, then reduce in graph order.
fn identify_unique_adjacency_masks<T, F>(
    posterior: &GraphPosterior,
    ctx: &ExecutionContext,
    identify: F,
) -> Result<std::collections::HashMap<AdjacencyMask, T>, CausalError>
where
    T: Send,
    F: Fn(&AdjacencyMask, &ExecutionContext) -> Result<T, CausalError> + Sync,
{
    let mut unique: Vec<&AdjacencyMask> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for mask in posterior.atoms().iter() {
        if seen.insert(mask) {
            unique.push(mask);
        }
    }
    let values = ctx.map_indexed(unique.len(), |i, inner| identify(unique[i], inner))?;
    Ok(unique.into_iter().cloned().zip(values).collect())
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
        return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_IDENTIFY });
    }
    ctx.map_indexed(posterior.n_graphs(), |i, inner| {
        if inner.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_IDENTIFY });
        }
        f(i, inner)
    })
}

/// Walk the samples of a static graph posterior once, recording each sample's key, weight
/// and identification flag, and collecting one atom per distinct graph key.
///
/// A posterior may list the same graph more than once (one entry per sample). Every entry
/// keeps its own weight and flag in the returned samples, but consumers weight an atom by the
/// combined mass of its key, so each key contributes exactly one atom: `classify` is told
/// whether the sample is the first of its key and is the only place an atom is built, which
/// keeps a repeated sample from cloning an identification it will not use.
fn collect_graph_posterior_atoms<A>(
    posterior: &GraphPosterior,
    ctx: &ExecutionContext,
    mut classify: impl FnMut(usize, bool) -> Result<(GraphIdentFlag, Option<A>), CausalError>,
) -> Result<(WeightedGraphSamples, Vec<A>), CausalError> {
    use std::collections::hash_map::Entry;

    let mut weights = Vec::with_capacity(posterior.n_graphs());
    let mut flags = Vec::with_capacity(posterior.n_graphs());
    let mut keys = Vec::with_capacity(posterior.n_graphs());
    let mut atoms = Vec::new();
    let mut atom_masks: std::collections::HashMap<u64, AdjacencyMask> =
        std::collections::HashMap::new();

    for i in 0..posterior.n_graphs() {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_IDENTIFY });
        }
        if let Some(progress) = &ctx.progress {
            progress.report(i as f64 / posterior.n_graphs().max(1) as f64, "envelope.identify");
        }
        let mask = &posterior.atoms()[i];
        let key = posterior.graph_keys()[i];
        keys.push(key);
        weights.push(posterior.weights()[i]);
        let first_for_key = match atom_masks.entry(key) {
            Entry::Vacant(slot) => {
                slot.insert(mask.clone());
                true
            }
            Entry::Occupied(slot) if slot.get() == mask => false,
            Entry::Occupied(_) => {
                return Err(CausalError::Compile {
                    message: "graph posterior reuses one graph key for different adjacency masks"
                        .into(),
                });
            }
        };
        let (flag, atom) = classify(i, first_for_key)?;
        flags.push(flag);
        atoms.extend(atom);
    }
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_IDENTIFY });
    }
    Ok((WeightedGraphSamples::new(weights, flags, keys)?, atoms))
}

/// The identified atoms of a DAG/ADMG posterior by adjacency mask, as
/// [`identify_unique_adjacency_masks`] returns them.
type MaskIdentifications =
    std::collections::HashMap<AdjacencyMask, Option<(IdentifiedEstimand, IdentificationResult)>>;

/// [`collect_graph_posterior_atoms`] over per-mask identifications: a sample is identified
/// when its mask is, and the first sample of a key contributes the atom.
fn collect_mask_atoms(
    posterior: &GraphPosterior,
    ctx: &ExecutionContext,
    by_mask: &MaskIdentifications,
) -> Result<CachedGraphPosteriorIdentification, CausalError> {
    let (graphs, atoms) = collect_graph_posterior_atoms(posterior, ctx, |i, first_for_key| {
        let Some(Some((estimand, identification))) = by_mask.get(&posterior.atoms()[i]) else {
            return Ok((GraphIdentFlag::Unidentified, None));
        };
        let atom = first_for_key.then(|| CachedGraphPosteriorAtomIdentification {
            key: posterior.graph_keys()[i],
            estimand: estimand.clone(),
            identification: identification.clone(),
            transport: None,
        });
        Ok((GraphIdentFlag::Identified, atom))
    })?;
    Ok(CachedGraphPosteriorIdentification {
        graphs,
        atoms: Arc::from(atoms),
        class_atoms: Arc::from([]),
    })
}

fn dbn_envelope_key(index: usize) -> Result<u64, CausalError> {
    u64::try_from(index).map_err(|_| CausalError::Compile {
        message: "DBN posterior has too many atoms for envelope keys".into(),
    })
}
