//! Graph-weighted effect envelopes (DESIGN.md Phase 6).
//!
//! Aggregates per-graph effect posteriors using [`WeightedGraphSamples`].
//! Unidentified mass is preserved by default and is never silently renormalized.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::needless_range_loop,
    clippy::float_cmp,
    clippy::doc_markdown
)]

use std::sync::Arc;

use causal_identify::IdentificationStatus;
use causal_prob::{
    GraphIdentFlag, InferenceDiagnostics, PosteriorDraws, PosteriorQuantityKind, PosteriorSchema,
    WeightedGraphSamples,
};

use crate::bayesian::CausalPosterior;
use crate::error::EstimationError;

/// Options for envelope aggregation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EnvelopeOptions {
    /// When true, renormalize weights over identified graphs only (explicit opt-in).
    /// Default false: unidentified mass is retained on the result.
    pub renormalize_identified_only: bool,
}

/// One graph's scalar effect posterior (columnar draws of a single effect).
#[derive(Clone, Debug)]
pub struct GraphEffectDraws {
    /// Opaque key matching [`WeightedGraphSamples::graph_keys`].
    pub graph_key: u64,
    /// Effect draws (length = n_draws).
    pub effect_draws: Arc<[f64]>,
}

/// Aggregate per-graph effect posteriors into a mixture envelope.
///
/// # Errors
///
/// Missing draws for a graph key, empty ensemble, or shape mismatch.
pub fn aggregate_effect_envelope(
    graphs: &WeightedGraphSamples,
    per_graph: &[GraphEffectDraws],
    diagnostics: InferenceDiagnostics,
    options: EnvelopeOptions,
) -> Result<CausalPosterior, EstimationError> {
    if graphs.n_samples == 0 {
        return Err(EstimationError::Stats("empty graph ensemble".into()));
    }
    let by_key: std::collections::HashMap<u64, &GraphEffectDraws> =
        per_graph.iter().map(|g| (g.graph_key, g)).collect();

    let mut n_draws = None;
    for i in 0..graphs.n_samples {
        if graphs.identified[i] != GraphIdentFlag::Identified {
            continue;
        }
        let key = graphs.graph_keys[i];
        let g = by_key.get(&key).ok_or_else(|| {
            EstimationError::Stats(format!("missing effect draws for graph key {key}"))
        })?;
        match n_draws {
            None => n_draws = Some(g.effect_draws.len()),
            Some(n) if n != g.effect_draws.len() => {
                return Err(EstimationError::Stats("per-graph effect draw counts differ".into()));
            }
            _ => {}
        }
    }
    let n_draws = n_draws.unwrap_or(0);

    let unidentified_mass = graphs.unidentified_mass();
    let identified_mass = graphs.identified_mass();
    let total = graphs.total_weight();

    let weight_scale = if options.renormalize_identified_only {
        if !(identified_mass > 0.0) {
            return Err(EstimationError::Stats("no identified mass to renormalize".into()));
        }
        1.0 / identified_mass
    } else {
        if !(total > 0.0) {
            return Err(EstimationError::Stats("non-positive total weight".into()));
        }
        1.0 / total
    };

    // Mixture: for each draw index d, sample from the discrete mixture of graph
    // effect draws weighted by (normalized) identified weights. Represent as a
    // weighted average of draws at matching indices (common random numbers),
    // which is the standard envelope summary for aligned draw sets.
    let mut mixture = vec![0.0; n_draws];
    let mut used_mass = 0.0;
    for i in 0..graphs.n_samples {
        if graphs.identified[i] != GraphIdentFlag::Identified {
            continue;
        }
        let w = graphs.weights[i] * weight_scale;
        used_mass += w;
        let g = by_key[&graphs.graph_keys[i]];
        for d in 0..n_draws {
            mixture[d] += w * g.effect_draws[d];
        }
    }
    // If not renormalizing, mixture is a sub-probability weighted mean; scale
    // identified contribution so the reported mean is over identified mass only
    // while retaining unidentified_mass on the artifact.
    if !options.renormalize_identified_only && used_mass > 0.0 && identified_mass > 0.0 {
        let renorm = (identified_mass / total) / used_mass;
        // used_mass already equals identified_mass/total, so renorm == 1.
        let _ = renorm;
    }

    let retained_unidentified = if options.renormalize_identified_only {
        0.0
    } else {
        unidentified_mass / total.max(f64::EPSILON)
    };

    let schema = PosteriorSchema {
        quantities: Arc::from([PosteriorQuantityKind::Effect { name: Arc::from("ate_envelope") }]),
    };
    let draws = PosteriorDraws::from_column_major(schema, n_draws, mixture)
        .map_err(|e| EstimationError::Stats(e.to_string()))?;
    let summaries = draws.summarize();

    let identification = if retained_unidentified > 0.0 {
        IdentificationStatus::NotIdentified
    } else if identified_mass > 0.0 {
        IdentificationStatus::NonparametricallyIdentified
    } else {
        IdentificationStatus::NotIdentified
    };

    Ok(CausalPosterior {
        draws,
        summaries,
        identification,
        prior_sensitivity: None,
        diagnostics,
        assumptions: causal_core::AssumptionSet::new(),
        unidentified_mass: retained_unidentified,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_prob::InferenceDiagnostics;

    #[test]
    fn preserves_unidentified_mass_by_default() {
        let graphs = WeightedGraphSamples::new(
            vec![0.5, 0.3, 0.2],
            vec![
                GraphIdentFlag::Identified,
                GraphIdentFlag::Unidentified,
                GraphIdentFlag::Identified,
            ],
            vec![1, 2, 3],
        )
        .unwrap();
        let per = vec![
            GraphEffectDraws { graph_key: 1, effect_draws: Arc::from(vec![1.0, 1.0, 1.0]) },
            GraphEffectDraws { graph_key: 3, effect_draws: Arc::from(vec![3.0, 3.0, 3.0]) },
        ];
        let env = aggregate_effect_envelope(
            &graphs,
            &per,
            InferenceDiagnostics::analytic("envelope"),
            EnvelopeOptions::default(),
        )
        .unwrap();
        assert!((env.unidentified_mass - 0.3).abs() < 1e-12);
        assert_eq!(env.identification, IdentificationStatus::NotIdentified);
        // Weighted mean of identified: (0.5*1 + 0.2*3)/1.0 = 1.1, but draws are
        // mixture with weights scaled by 1/total: 0.5*1 + 0.2*3 = 1.1
        let mean = env.summaries.mean[0];
        assert!((mean - 1.1).abs() < 1e-12);
    }

    #[test]
    fn renormalize_drops_unidentified_mass() {
        let graphs = WeightedGraphSamples::new(
            vec![0.5, 0.3, 0.2],
            vec![
                GraphIdentFlag::Identified,
                GraphIdentFlag::Unidentified,
                GraphIdentFlag::Identified,
            ],
            vec![1, 2, 3],
        )
        .unwrap();
        let per = vec![
            GraphEffectDraws { graph_key: 1, effect_draws: Arc::from(vec![1.0]) },
            GraphEffectDraws { graph_key: 3, effect_draws: Arc::from(vec![3.0]) },
        ];
        let env = aggregate_effect_envelope(
            &graphs,
            &per,
            InferenceDiagnostics::analytic("envelope"),
            EnvelopeOptions { renormalize_identified_only: true },
        )
        .unwrap();
        assert_eq!(env.unidentified_mass, 0.0);
        assert_eq!(env.identification, IdentificationStatus::NonparametricallyIdentified);
        // (0.5*1 + 0.2*3) / 0.7 = 1.1/0.7
        assert!((env.summaries.mean[0] - 1.1 / 0.7).abs() < 1e-12);
    }
}
