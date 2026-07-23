//! Graph-weighted effect envelopes.
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

use causal_core::IdentificationStatus;
use causal_prob::{
    GraphIdentFlag, InferenceDiagnostics, PosteriorDraws, PosteriorQuantityKind, PosteriorSchema,
    WeightedGraphSamples,
};

use crate::bayesian::CausalPosterior;
use crate::error::EstimationError;

/// Options for envelope aggregation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EnvelopeOptions {
    /// When true, drop unidentified mass from the artifact (identification may
    /// become fully identified). Draws always use E[τ | identified] either way.
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
        return Err(EstimationError::stats_msg("empty graph ensemble"));
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
            EstimationError::stats_msg(format!("missing effect draws for graph key {key}"))
        })?;
        match n_draws {
            None => n_draws = Some(g.effect_draws.len()),
            Some(n) if n != g.effect_draws.len() => {
                return Err(EstimationError::stats_msg("per-graph effect draw counts differ"));
            }
            _ => {}
        }
    }
    let n_draws = n_draws.unwrap_or(0);

    let unidentified_mass = graphs.unidentified_mass();
    let identified_mass = graphs.identified_mass();
    let total = graphs.total_weight();

    // Draws always report E[τ | identified]: normalize over identified mass.
    // Unidentified mass is retained on the artifact unless explicitly dropped.
    if !(identified_mass > 0.0) {
        return Err(EstimationError::stats_msg("no identified mass for effect envelope"));
    }
    if !(total > 0.0) {
        return Err(EstimationError::stats_msg("non-positive total weight"));
    }
    let weight_scale = 1.0 / identified_mass;

    // Mixture: weighted average of aligned per-graph draws (common random numbers).
    let mut mixture = vec![0.0; n_draws];
    for i in 0..graphs.n_samples {
        if graphs.identified[i] != GraphIdentFlag::Identified {
            continue;
        }
        let w = graphs.weights[i] * weight_scale;
        let g = by_key[&graphs.graph_keys[i]];
        for d in 0..n_draws {
            mixture[d] += w * g.effect_draws[d];
        }
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
        .map_err(EstimationError::from)?;
    let summaries = draws.summarize();

    let identification = if identified_mass > 0.0 && retained_unidentified > 0.0 {
        IdentificationStatus::GraphDependent
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
        conflict_summary: None,
        diagnostics,
        assumptions: causal_core::AssumptionSet::new(),
        unidentified_mass: retained_unidentified,
        early_stopped: false,
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
        assert_eq!(env.identification, IdentificationStatus::GraphDependent);
        // E[τ | identified] = (0.5*1 + 0.2*3) / 0.7
        let mean = env.summaries.mean[0];
        assert!((mean - 1.1 / 0.7).abs() < 1e-12);
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

    #[test]
    fn interactive_subsample_mass_accounting_honest() {
        use causal_core::CausalRng;

        let graphs = WeightedGraphSamples::new(
            vec![0.2, 0.2, 0.2, 0.2, 0.2],
            vec![
                GraphIdentFlag::Identified,
                GraphIdentFlag::Identified,
                GraphIdentFlag::Identified,
                GraphIdentFlag::Identified,
                GraphIdentFlag::Unidentified,
            ],
            vec![1, 2, 3, 4, 5],
        )
        .unwrap();
        let per_full: Vec<GraphEffectDraws> = [1u64, 2, 3, 4]
            .into_iter()
            .map(|k| GraphEffectDraws { graph_key: k, effect_draws: Arc::from(vec![k as f64; 4]) })
            .collect();
        let full = aggregate_effect_envelope(
            &graphs,
            &per_full,
            InferenceDiagnostics::analytic("full"),
            EnvelopeOptions::default(),
        )
        .unwrap();
        assert!((full.unidentified_mass - 0.2).abs() < 1e-12);

        let mut rng = CausalRng::from_seed(3);
        let sub = graphs.stratified_interactive_subsample(2, &mut rng).unwrap();
        assert!(sub.approximate);
        let keep: std::collections::HashSet<u64> = sub
            .graphs
            .graph_keys
            .iter()
            .zip(sub.graphs.identified.iter())
            .filter(|(_, f)| **f == GraphIdentFlag::Identified)
            .map(|(k, _)| *k)
            .collect();
        let per_sub: Vec<_> =
            per_full.into_iter().filter(|g| keep.contains(&g.graph_key)).collect();
        let approx = aggregate_effect_envelope(
            &sub.graphs,
            &per_sub,
            InferenceDiagnostics::analytic("approx"),
            EnvelopeOptions::default(),
        )
        .unwrap();
        // Mass honesty: unidentified = original UID + leftover identified, / total.
        let expected_uid =
            (graphs.unidentified_mass() + sub.leftover_identified_mass) / graphs.total_weight();
        let mass_err = (approx.unidentified_mass - expected_uid).abs();
        assert!(mass_err < 1e-12);
        assert!(approx.unidentified_mass > full.unidentified_mass);
        // Mean is E[τ | identified-in-subset], not silently the full mixture.
        assert!(approx.summaries.mean[0].is_finite());
    }
}
