//! Graph-weighted effect envelopes.
//!
//! Aggregates per-graph effect posteriors using [`WeightedGraphSamples`].
//! Published moments and draws are the identified-atom BMA (`P(τ | identified)`).
//! Unidentified mass is a separate, non-renormalized axis on the result and is
//! never redistributed into that effect posterior. The aggregators see only
//! identification flags, so atoms a latency tier flagged out of a subsample land
//! in `unidentified_mass` here; callers that subsample move that share to
//! [`CausalPosterior::subsampled_out_mass`].
//! [`aggregate_mixture_functional_envelope`] instead publishes the posterior of
//! the frozen-weight mixture functional, for equivalence-class envelopes whose
//! weights the data cannot update.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::needless_range_loop,
    clippy::float_cmp,
    clippy::doc_markdown,
    clippy::too_many_lines
)]

use std::sync::Arc;

use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, CausalRng, IdentificationStatus, ParametricAssumption,
};
use antecedent_prob::{
    GraphIdentFlag, InferenceDiagnostics, PosteriorDraws, PosteriorQuantityKind, PosteriorSchema,
    WeightedGraphSamples,
};

use crate::bayesian::CausalPosterior;
use crate::error::EstimationError;

/// Options for envelope aggregation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EnvelopeOptions {
    /// Request to publish a full-support mixture after dropping unidentified
    /// mass. That claim is refused when unidentified mass is present: the
    /// published effect stays `P(τ | identified)` and unidentified mass remains
    /// a separate axis. Default `false` keeps that honest split.
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
/// Invalid weights, missing/duplicate draws, non-finite or empty draws,
/// incompatible shapes, or unrepresentable moments.
///
/// Moments are exact for the supplied empirical component posteriors (up to
/// floating-point error). Quantiles use reproducible finite Monte Carlo draws;
/// they are not exact quantiles of the weighted empirical mixture.
pub fn aggregate_effect_envelope(
    graphs: &WeightedGraphSamples,
    per_graph: &[GraphEffectDraws],
    diagnostics: InferenceDiagnostics,
    options: EnvelopeOptions,
) -> Result<CausalPosterior, EstimationError> {
    if graphs.n_samples == 0 {
        return Err(EstimationError::stats_msg("empty graph ensemble"));
    }
    if graphs.weights.len() != graphs.n_samples
        || graphs.identified.len() != graphs.n_samples
        || graphs.graph_keys.len() != graphs.n_samples
        || graphs.weights.iter().any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err(EstimationError::stats_msg("invalid graph ensemble shape or weights"));
    }
    let mut by_key = std::collections::HashMap::new();
    for graph in per_graph {
        if by_key.insert(graph.graph_key, graph).is_some() {
            return Err(EstimationError::stats_msg("duplicate per-graph effect draws"));
        }
    }
    let total = graphs.total_weight();
    if !total.is_finite() || total <= 0.0 {
        return Err(EstimationError::stats_msg("non-positive or non-finite total weight"));
    }
    let unidentified_mass = graphs.unidentified_mass();
    // Sum directly: total - unidentified loses tiny positive identified mass.
    let identified_mass: f64 = graphs
        .weights
        .iter()
        .zip(graphs.identified.iter())
        .filter(|(_, flag)| **flag == GraphIdentFlag::Identified)
        .map(|(weight, _)| *weight)
        .sum();
    if identified_mass <= 0.0 {
        return Err(EstimationError::stats_msg("no identified mass for effect envelope"));
    }
    if options.renormalize_identified_only && unidentified_mass > 0.0 {
        return Err(EstimationError::stats_msg(
            "renormalize_identified_only refuses to publish a 100% mixture after dropping \
             unidentified mass; E[τ] and draws stay P(τ | identified) and unidentified mass \
             is retained as a separate axis",
        ));
    }
    let retained_unidentified = unidentified_mass / total;
    let mut n_draws = None;
    for i in 0..graphs.n_samples {
        if graphs.identified[i] != GraphIdentFlag::Identified || graphs.weights[i] == 0.0 {
            continue;
        }
        let key = graphs.graph_keys[i];
        let graph = by_key.get(&key).ok_or_else(|| {
            EstimationError::stats_msg(format!("missing effect draws for graph key {key}"))
        })?;
        if graph.effect_draws.is_empty() || graph.effect_draws.iter().any(|x| !x.is_finite()) {
            return Err(EstimationError::stats_msg("effect draws must be nonempty and finite"));
        }
        match n_draws {
            None => n_draws = Some(graph.effect_draws.len()),
            Some(n) if n != graph.effect_draws.len() => {
                return Err(EstimationError::stats_msg("per-graph effect draw counts differ"));
            }
            _ => {}
        }
    }
    let n_draws = n_draws.unwrap_or(0);

    // Canonicalize by graph key so the posterior is invariant to atom input order.
    // Repeated graph keys (for example, repeated MCMC samples) contribute their
    // combined posterior mass.
    let mut identified_atoms = Vec::<(u64, f64)>::new();
    for i in 0..graphs.n_samples {
        if graphs.identified[i] != GraphIdentFlag::Identified || graphs.weights[i] == 0.0 {
            continue;
        }
        identified_atoms.push((graphs.graph_keys[i], graphs.weights[i]));
    }
    identified_atoms.sort_by_key(|(key, _)| *key);
    let mut combined = Vec::<(u64, f64)>::with_capacity(identified_atoms.len());
    for (key, weight) in identified_atoms {
        match combined.last_mut() {
            Some((last_key, last_weight)) if *last_key == key => *last_weight += weight,
            _ => combined.push((key, weight)),
        }
    }

    // Ordinary Bayesian model averaging is a mixture distribution: each draw
    // first samples the graph model, then samples that graph's effect posterior.
    // Averaging aligned draw indices across graphs would impose an arbitrary
    // cross-model coupling and erase between-graph variance.
    let mut cumulative = 0.0;
    let cdf = combined
        .iter()
        .map(|(_, weight)| {
            cumulative += weight / identified_mass;
            cumulative
        })
        .collect::<Vec<_>>();
    // A fixed stream preserves this API's deterministic, atom-order-invariant
    // contract. These are pseudo-random mixture draws, not aligned averages or
    // deterministic thinning of potentially ordered within-model draws.
    let mut rng = CausalRng::from_seed(0x454E_5645_4C4F_5045);
    let mut mixture = Vec::with_capacity(n_draws);
    for _ in 0..n_draws {
        let target = rng.next_f64();
        let atom = cdf.partition_point(|mass| *mass <= target).min(combined.len() - 1);
        let graph_draws = &by_key[&combined[atom].0].effect_draws;
        let index = (rng.next_f64() * n_draws as f64) as usize;
        mixture.push(graph_draws[index.min(n_draws - 1)]);
    }

    // Exact moments of the supplied empirical component posteriors, conditional
    // on identification. Center before summing to avoid E[X²] - E[X]²
    // cancellation for large offsets with small within/between-model variation.
    let origin = by_key[&combined[0].0].effect_draws[0];
    let centered_mean = combined
        .iter()
        .map(|(key, weight)| {
            let values = &by_key[key].effect_draws;
            (weight / identified_mass)
                * values.iter().map(|x| (x - origin) / n_draws as f64).sum::<f64>()
        })
        .sum::<f64>();
    let exact_mean = origin + centered_mean;
    let variance = combined
        .iter()
        .map(|(key, weight)| {
            let values = &by_key[key].effect_draws;
            (weight / identified_mass)
                * values
                    .iter()
                    .map(|x| ((x - origin) - centered_mean).powi(2) / n_draws as f64)
                    .sum::<f64>()
        })
        .sum::<f64>();
    let exact_sd = variance.sqrt();
    if !exact_mean.is_finite() || !exact_sd.is_finite() {
        return Err(EstimationError::stats_msg("effect mixture moments overflow"));
    }

    let schema = PosteriorSchema {
        quantities: Arc::from([PosteriorQuantityKind::Effect { name: Arc::from("ate_envelope") }]),
    };
    let draws = PosteriorDraws::from_column_major(schema, n_draws, mixture)
        .map_err(EstimationError::from)?;
    let mut summaries = draws.summarize();
    summaries.mean = Arc::from([exact_mean]);
    summaries.sd = Arc::from([exact_sd]);

    let identification = if identified_mass > 0.0 && retained_unidentified > 0.0 {
        IdentificationStatus::GraphDependent
    } else if identified_mass > 0.0 {
        IdentificationStatus::NonparametricallyIdentified
    } else {
        IdentificationStatus::NotIdentified
    };

    let mut assumptions = AssumptionSet::new();
    assumptions.push(AssumptionRecord {
        assumption: Assumption::ParametricRestriction(ParametricAssumption {
            id: Arc::from("envelope.conditional_on_identification"),
            description: Arc::from(
                "Published effect moments and draws are E[τ | identified] (identified-atom BMA). \
                 Unidentified mass is retained as a separate, non-renormalized axis and is not \
                 redistributed into the effect mixture. This is not a full-support posterior.",
            ),
        }),
        source: AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("estimate.aggregate_effect_envelope"),
        },
        scope: AssumptionScope::Estimation,
        status: AssumptionStatus::Declared,
    });

    Ok(CausalPosterior {
        subsampled_out_mass: 0.0,
        unevaluable_mass: 0.0,
        draws,
        summaries,
        identification,
        prior_sensitivity: None,
        conflict_summary: None,
        diagnostics,
        assumptions,
        unidentified_mass: retained_unidentified,
        early_stopped: false,
        treatment_contrast: None,
    })
}

/// Posterior of the frozen-weight mixture functional `Σ_g w̄_g τ_g` over
/// identified envelope atoms that were fitted on the same rows.
///
/// [`aggregate_effect_envelope`] publishes the identified-atom BMA: each draw
/// first picks a completion, so its spread includes the disagreement *between*
/// completions. That is the right object when the weights are data-updated
/// graph probabilities. For a Markov-equivalence envelope the weights are a
/// frozen enumeration the data cannot update, and the reported number is the
/// mixture functional itself; its interval must describe uncertainty in that
/// functional, not the spread of completion-specific effects.
///
/// Draws are coupled across atoms with an empirical Gaussian copula (rank
/// matching, Iman–Conover): correlated standard normals with the atoms'
/// sampling correlation (`correlation`, row-major `k × k` in `per_graph`
/// order, e.g. from their influence functions on the shared rows) fix the
/// joint ranks, and each atom's own posterior draws are permuted onto those
/// ranks. Every atom keeps its exact marginal posterior (prior included);
/// identical estimands (correlation 1) move together. The mean equals the
/// BMA mean exactly; the SD is the empirical SD of the coupled mixture draws.
/// Unidentified mass stays a separate, non-renormalized axis.
///
/// # Errors
///
/// Everything [`aggregate_effect_envelope`] refuses, a correlation matrix of
/// the wrong size, or non-finite / out-of-range correlations.
pub fn aggregate_mixture_functional_envelope(
    graphs: &WeightedGraphSamples,
    per_graph: &[GraphEffectDraws],
    correlation: &[f64],
    diagnostics: InferenceDiagnostics,
) -> Result<CausalPosterior, EstimationError> {
    let mut posterior =
        aggregate_effect_envelope(graphs, per_graph, diagnostics, EnvelopeOptions::default())?;
    let k = per_graph.len();
    if correlation.len() != k * k
        || correlation.iter().any(|r| !r.is_finite() || r.abs() > 1.0 + 1e-9)
    {
        return Err(EstimationError::stats_msg(
            "mixture-functional coupling needs a finite k×k correlation over the fitted atoms",
        ));
    }
    let mut weights = vec![0.0; k];
    for i in 0..graphs.n_samples {
        if graphs.identified[i] != GraphIdentFlag::Identified {
            continue;
        }
        if let Some(pos) = per_graph.iter().position(|g| g.graph_key == graphs.graph_keys[i]) {
            weights[pos] += graphs.weights[i];
        }
    }
    let mass: f64 = weights.iter().sum();
    if mass <= 0.0 {
        return Err(EstimationError::stats_msg("no identified mass for effect envelope"));
    }
    let atoms: Vec<(f64, &[f64])> = weights
        .iter()
        .zip(per_graph)
        .map(|(w, g)| (w / mass, g.effect_draws.as_ref()))
        .filter(|(w, _)| *w > 0.0)
        .collect();
    let kept: Vec<usize> = (0..k).filter(|&g| weights[g] > 0.0).collect();
    let sub: Vec<f64> =
        kept.iter().flat_map(|&a| kept.iter().map(move |&b| correlation[a * k + b])).collect();
    let mixture = couple_mixture_functional_draws(&atoms, &sub, 0x4D49_5846_554E_4354)?;
    let n = mixture.len() as f64;
    let mean = posterior.summaries.mean[0];
    let sd = (mixture.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)).sqrt();
    let schema = posterior.draws.schema.clone();
    posterior.draws = PosteriorDraws::from_column_major(schema, mixture.len(), mixture)
        .map_err(EstimationError::from)?;
    let mut summaries = posterior.draws.summarize();
    summaries.mean = Arc::from([mean]);
    summaries.sd = Arc::from([sd]);
    posterior.summaries = summaries;
    let mut assumptions = AssumptionSet::new();
    assumptions.push(AssumptionRecord {
        assumption: Assumption::ParametricRestriction(ParametricAssumption {
            id: Arc::from("envelope.mixture_functional"),
            description: Arc::from(
                "Published effect draws are the posterior of the frozen-weight mixture \
                 functional sum_g w_g tau_g over identified completions (each completion's \
                 own posterior, coupled across completions by their sampling correlation on \
                 the shared rows). They are not a distribution over completion-specific \
                 effects. Unidentified mass is retained as a separate, non-renormalized axis.",
            ),
        }),
        source: AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("estimate.aggregate_mixture_functional_envelope"),
        },
        scope: AssumptionScope::Estimation,
        status: AssumptionStatus::Declared,
    });
    posterior.assumptions = assumptions;
    Ok(posterior)
}

/// Rank-coupled mixture draws `Σ_g w_g τ_g^(d)` (weights already normalized).
///
/// # Errors
///
/// Empty or ragged draws, or a correlation of the wrong size.
#[allow(clippy::many_single_char_names)]
pub fn couple_mixture_functional_draws(
    atoms: &[(f64, &[f64])],
    correlation: &[f64],
    seed: u64,
) -> Result<Vec<f64>, EstimationError> {
    let k = atoms.len();
    if k == 0 || correlation.len() != k * k {
        return Err(EstimationError::stats_msg("coupling needs atoms and a k×k correlation"));
    }
    let n = atoms[0].1.len();
    if n == 0 || atoms.iter().any(|(_, d)| d.len() != n) {
        return Err(EstimationError::stats_msg("coupled atoms need equal, nonempty draw counts"));
    }
    // Positive-semidefinite Cholesky: a zero pivot marks an atom that is a
    // linear function of earlier ones (e.g. an identical estimand).
    let mut l = vec![0.0; k * k];
    for j in 0..k {
        let mut s = correlation[j * k + j];
        for p in 0..j {
            s -= l[j * k + p] * l[j * k + p];
        }
        let pivot = if s > 1e-12 { s.sqrt() } else { 0.0 };
        l[j * k + j] = pivot;
        for i in (j + 1)..k {
            let mut v = correlation[i * k + j];
            for p in 0..j {
                v -= l[i * k + p] * l[j * k + p];
            }
            l[i * k + j] = if pivot > 0.0 { v / pivot } else { 0.0 };
        }
    }
    let mut rng = CausalRng::from_seed(seed);
    let mut normal = move || {
        let u1 = rng.next_f64().max(1e-300);
        let u2 = rng.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    let mut latent = vec![0.0; n * k];
    let mut e = vec![0.0; k];
    for d in 0..n {
        for x in &mut e {
            *x = normal();
        }
        for i in 0..k {
            latent[d * k + i] = (0..=i).map(|p| l[i * k + p] * e[p]).sum();
        }
    }
    let mut mixture = vec![0.0; n];
    let mut order: Vec<usize> = (0..n).collect();
    for (g, (w, draws)) in atoms.iter().enumerate() {
        let mut sorted = draws.to_vec();
        sorted.sort_by(f64::total_cmp);
        order.sort_by(|&a, &b| latent[a * k + g].total_cmp(&latent[b * k + g]));
        for (rank, &d) in order.iter().enumerate() {
            mixture[d] += w * sorted[rank];
        }
    }
    Ok(mixture)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_prob::InferenceDiagnostics;

    #[test]
    fn coupled_mixture_keeps_marginals_and_tracks_correlation() {
        let n: usize = 4000;
        let a: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) / n as f64 - 0.5).collect();
        let b: Vec<f64> = a.iter().map(|x| 3.0 * x + 1.0).collect();
        // Perfect coupling: the mixture is the quantile average.
        let perfect =
            couple_mixture_functional_draws(&[(0.5, &a), (0.5, &b)], &[1.0, 1.0, 1.0, 1.0], 7)
                .unwrap();
        let mut sorted = perfect.clone();
        sorted.sort_by(f64::total_cmp);
        for (i, v) in sorted.iter().enumerate() {
            assert!((v - 0.5 * (a[i] + b[i])).abs() < 1e-12);
        }
        // Independence shrinks the spread; perfect coupling does not.
        let sd = |v: &[f64]| {
            let m = v.iter().sum::<f64>() / v.len() as f64;
            (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
        };
        let independent =
            couple_mixture_functional_draws(&[(0.5, &a), (0.5, &b)], &[1.0, 0.0, 0.0, 1.0], 7)
                .unwrap();
        assert!(sd(&independent) < 0.8 * sd(&perfect));
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        assert!((mean(&independent) - mean(&perfect)).abs() < 1e-12);
    }

    fn aggregate(
        weights: Vec<f64>,
        values: Vec<Vec<f64>>,
    ) -> Result<CausalPosterior, EstimationError> {
        let keys = (0..weights.len() as u64).collect::<Vec<_>>();
        let mut graphs = WeightedGraphSamples::new(
            vec![1.0; weights.len()],
            vec![GraphIdentFlag::Identified; keys.len()],
            keys,
        )
        .unwrap();
        // Public fields can bypass constructor validation. Exercise the
        // aggregator's own validation, including for malformed ensembles.
        graphs.weights = weights.into();
        let per = values
            .into_iter()
            .enumerate()
            .map(|(key, values)| GraphEffectDraws {
                graph_key: key as u64,
                effect_draws: values.into(),
            })
            .collect::<Vec<_>>();
        aggregate_effect_envelope(
            &graphs,
            &per,
            InferenceDiagnostics::analytic("test"),
            EnvelopeOptions::default(),
        )
    }

    #[test]
    fn large_location_does_not_erase_small_mixture_variance() {
        let posterior =
            aggregate(vec![0.5, 0.5], vec![vec![1e12 - 1.0; 128], vec![1e12 + 1.0; 128]]).unwrap();
        assert_eq!(posterior.summaries.mean[0], 1e12);
        assert_eq!(posterior.summaries.sd[0], 1.0);
    }

    #[test]
    fn categorical_draws_do_not_alias_ordered_component_samples() {
        let alternating = (0..4096).map(|i| if i % 2 == 0 { -1.0 } else { 1.0 }).collect();
        let posterior = aggregate(vec![0.5, 0.5], vec![alternating, vec![0.0; 4096]]).unwrap();
        let negatives = posterior.draws.values.iter().filter(|&&x| x < 0.0).count();
        let positives = posterior.draws.values.iter().filter(|&&x| x > 0.0).count();
        assert!((900..1150).contains(&negatives), "negative mass: {negatives}");
        assert!((900..1150).contains(&positives), "positive mass: {positives}");
    }

    #[test]
    fn invalid_weights_and_draws_are_errors() {
        for weight in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(aggregate(vec![weight, 1.0], vec![vec![1.0], vec![2.0]]).is_err());
        }
        for values in [vec![], vec![f64::NAN], vec![f64::INFINITY]] {
            assert!(aggregate(vec![1.0], vec![values]).is_err());
        }
        // A zero-probability atom needs no numerical posterior.
        assert!(aggregate(vec![1.0, 0.0], vec![vec![2.0]]).is_ok());
    }

    #[test]
    fn tiny_weights_preserve_mass_and_conditional_moments() {
        let graphs = WeightedGraphSamples::new(
            vec![1e-30, 1e-30],
            vec![GraphIdentFlag::Identified, GraphIdentFlag::Unidentified],
            vec![1, 2],
        )
        .unwrap();
        let per = [GraphEffectDraws { graph_key: 1, effect_draws: Arc::from([2.0, 4.0]) }];
        let posterior = aggregate_effect_envelope(
            &graphs,
            &per,
            InferenceDiagnostics::analytic("tiny"),
            EnvelopeOptions::default(),
        )
        .unwrap();
        assert_eq!(posterior.unidentified_mass, 0.5);
        assert_eq!(posterior.summaries.mean[0], 3.0);
        let imbalanced = WeightedGraphSamples::new(
            vec![1e-30, 1.0],
            vec![GraphIdentFlag::Identified, GraphIdentFlag::Unidentified],
            vec![1, 2],
        )
        .unwrap();
        assert!(
            aggregate_effect_envelope(
                &imbalanced,
                &per,
                InferenceDiagnostics::analytic("tiny"),
                EnvelopeOptions::default()
            )
            .is_ok()
        );
    }

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
            GraphEffectDraws { graph_key: 1, effect_draws: Arc::from(vec![1.0; 4_096]) },
            GraphEffectDraws { graph_key: 3, effect_draws: Arc::from(vec![3.0; 4_096]) },
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
        // E[τ | identified] = (0.5*1 + 0.2*3) / 0.7 is a mixture functional.
        let mean = env.summaries.mean[0];
        assert!((mean - 1.1 / 0.7).abs() < 1e-12);
        assert!(env.assumptions.entries.iter().any(|a| {
            matches!(
                &a.assumption,
                antecedent_core::Assumption::ParametricRestriction(p)
                    if p.id.as_ref() == "envelope.conditional_on_identification"
            )
        }));
    }

    #[test]
    fn categorical_mixing_retains_between_graph_variance() {
        let n_draws = 4_096;
        let graphs = WeightedGraphSamples::new(
            vec![0.5, 0.5],
            vec![GraphIdentFlag::Identified, GraphIdentFlag::Identified],
            vec![11, 22],
        )
        .unwrap();
        let per = vec![
            GraphEffectDraws { graph_key: 11, effect_draws: Arc::from(vec![0.0; n_draws]) },
            GraphEffectDraws { graph_key: 22, effect_draws: Arc::from(vec![10.0; n_draws]) },
        ];
        let posterior = aggregate_effect_envelope(
            &graphs,
            &per,
            InferenceDiagnostics::analytic("categorical"),
            EnvelopeOptions::default(),
        )
        .unwrap();

        assert!((posterior.summaries.mean[0] - 5.0).abs() < 1e-12);
        assert!(
            posterior.summaries.sd[0] > 4.9,
            "categorical graph mixing must retain between-graph variance"
        );
    }

    #[test]
    fn categorical_mixing_is_atom_order_invariant() {
        let graphs_a = WeightedGraphSamples::new(
            vec![0.25, 0.75],
            vec![GraphIdentFlag::Identified, GraphIdentFlag::Identified],
            vec![1, 2],
        )
        .unwrap();
        let graphs_b = WeightedGraphSamples::new(
            vec![0.75, 0.25],
            vec![GraphIdentFlag::Identified, GraphIdentFlag::Identified],
            vec![2, 1],
        )
        .unwrap();
        let per_a = vec![
            GraphEffectDraws { graph_key: 1, effect_draws: Arc::from(vec![1.0, 2.0, 3.0]) },
            GraphEffectDraws { graph_key: 2, effect_draws: Arc::from(vec![7.0, 8.0, 9.0]) },
        ];
        let per_b = per_a.iter().cloned().rev().collect::<Vec<_>>();
        let a = aggregate_effect_envelope(
            &graphs_a,
            &per_a,
            InferenceDiagnostics::analytic("order-a"),
            EnvelopeOptions::default(),
        )
        .unwrap();
        let b = aggregate_effect_envelope(
            &graphs_b,
            &per_b,
            InferenceDiagnostics::analytic("order-b"),
            EnvelopeOptions::default(),
        )
        .unwrap();

        assert_eq!(a.draws.values.as_ref(), b.draws.values.as_ref());
        assert_eq!(a.unidentified_mass, b.unidentified_mass);
    }

    #[test]
    fn renormalize_identified_only_refuses_to_drop_mass() {
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
        let err = aggregate_effect_envelope(
            &graphs,
            &per,
            InferenceDiagnostics::analytic("envelope"),
            EnvelopeOptions { renormalize_identified_only: true },
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("unidentified"),
            "renormalize_identified_only must refuse when unidentified mass is present: {err}"
        );
    }

    #[test]
    fn interactive_subsample_mass_accounting_honest() {
        use antecedent_core::CausalRng;

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
        // Mass honesty: the aggregator only sees flags, so every atom outside
        // the mixture (original UID + leftover identified) is uncovered mass.
        // Study paths then split the leftover out as subsampled_out_mass.
        assert!(approx.subsampled_out_mass.abs() < f64::EPSILON);
        let expected_uid =
            (graphs.unidentified_mass() + sub.leftover_identified_mass) / graphs.total_weight();
        let mass_err = (approx.unidentified_mass - expected_uid).abs();
        assert!(mass_err < 1e-12);
        assert!(approx.unidentified_mass > full.unidentified_mass);
        // Mean is E[τ | identified-in-subset], not silently the full mixture.
        assert!(approx.summaries.mean[0].is_finite());
    }
}
