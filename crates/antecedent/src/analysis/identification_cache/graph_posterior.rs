//! Static graph-posterior identification caches: DAG, ADMG and CPDAG/PAG atoms.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, CausalQuery, ExecutionContext, Intervention, ResponseQuery, Value,
};
use antecedent_discovery::GraphPosterior;
use antecedent_discovery::adjacency_mask::dag_from_adjacency_mask;
use antecedent_expr::IdentifiedEstimand;
use antecedent_identify::{IdentificationEnvelope, IdentificationResult, IdentificationStatus};
use antecedent_prob::{GraphIdentFlag, WeightedGraphSamples};

use super::{
    collect_graph_posterior_atoms, collect_mask_atoms, identify_unique_adjacency_masks,
    map_posterior_graphs,
};
use crate::error::CausalError;

/// One identified atom in a prepared static graph posterior.
#[derive(Clone, Debug)]
pub(crate) struct CachedGraphPosteriorAtomIdentification {
    /// Opaque graph key retained by the posterior envelope.
    pub key: u64,
    /// Per-atom selected estimand.
    pub estimand: IdentifiedEstimand,
    /// Full per-atom identification result.
    pub identification: IdentificationResult,
    /// The certified sID transport proof of this atom's selection diagram, retained only
    /// by a `TransportQuery` posterior: the trial-to-target estimator reads its formula.
    pub transport: Option<Arc<antecedent_identify::TransportIdentification>>,
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
    /// [`Self::graphs`] (`IdentifiedKeyWeights`), which keeps one entry per
    /// posterior sample. Unidentified atoms remain in [`Self::graphs`] with
    /// [`GraphIdentFlag::Unidentified`].
    pub atoms: Arc<[CachedGraphPosteriorAtomIdentification]>,
    /// CPDAG/PAG posterior atoms evaluated with the class ATE envelope.
    /// Empty when [`antecedent_discovery::GraphPosterior::atom_kind`] is DAG or ADMG.
    pub class_atoms: Arc<[CachedClassPosteriorAtomIdentification]>,
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
    use crate::strategy_table::{
        DEFAULT_IDENTIFIER_ID, EstimatorId, identify_static_query_in_context, select_estimand,
    };

    // A DBN posterior's contemporaneous masks are valid DAGs, but identifying a
    // static effect on them alone would drop every lagged confounder. That is
    // a different coordinate (temporal Pulse / Sustained), so fail closed.
    if posterior.lag_masks().is_some() || posterior.max_lag().is_some() {
        return Err(CausalError::Unsupported {
            message: "static AverageEffect over a graph posterior requires static DAG atoms; \
                      this posterior carries DBN lag structure, so it needs a temporal \
                      Pulse / single-step Sustained query",
        });
    }
    match posterior.atom_kind() {
        antecedent_discovery::GraphPosteriorAtomKind::Dag => {}
        antecedent_discovery::GraphPosteriorAtomKind::Admg
        | antecedent_discovery::GraphPosteriorAtomKind::TemporalAdmg => {
            return build_admg_graph_posterior_identification_cache(posterior, query, ctx);
        }
        _ => return build_class_graph_posterior_identification_cache(posterior, query, ctx),
    }
    crate::analysis::execute::report_identify_compute(ctx);
    let by_mask = identify_unique_adjacency_masks(posterior, ctx, |mask, inner| {
        (|| -> Result<Option<(IdentifiedEstimand, IdentificationResult)>, CausalError> {
            let Ok(dag) = dag_from_adjacency_mask(mask, posterior.n_vars()) else {
                return Ok(None);
            };
            let Some(identification) = class_envelope_outcome(identify_static_query_in_context(
                DEFAULT_IDENTIFIER_ID,
                &dag,
                &CausalQuery::AverageEffect(query.clone()),
                None,
                Some(inner),
            ))?
            else {
                return Ok(None);
            };
            if !crate::analysis::execute::identification_status_ok_for_case(identification.status)
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
    collect_mask_atoms(posterior, ctx, &by_mask)
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
    use crate::strategy_table::{
        DEFAULT_ADMG_IDENTIFIER_ID, EstimatorId, identify_admg_query, select_estimand,
    };
    use antecedent_discovery::adjacency_mask::admg_from_adjacency_mask;

    crate::analysis::execute::report_identify_compute(ctx);
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
            let first_level = treatment.grid.values()?.into_iter().next().ok_or_else(|| {
                CausalError::Compile {
                    message: "MeanCurve response requires a non-empty evaluation grid".into(),
                }
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
            let Ok(admg) = admg_from_adjacency_mask(mask, posterior.n_vars()) else {
                return Ok(None);
            };
            let Some(identification) = class_envelope_outcome(identify_admg_query(
                DEFAULT_ADMG_IDENTIFIER_ID,
                &admg,
                &causal_query,
            ))?
            else {
                return Ok(None);
            };
            if !crate::analysis::execute::identification_status_ok_for_case(identification.status)
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
    collect_mask_atoms(posterior, ctx, &by_mask)
}

fn build_admg_graph_posterior_identification_cache(
    posterior: &GraphPosterior,
    query: &AverageEffectQuery,
    ctx: &ExecutionContext,
) -> Result<CachedGraphPosteriorIdentification, CausalError> {
    use crate::strategy_table::{
        DEFAULT_ADMG_IDENTIFIER_ID, EstimatorId, identify_admg, select_estimand,
    };
    use antecedent_discovery::adjacency_mask::admg_from_adjacency_mask;

    crate::analysis::execute::report_identify_compute(ctx);
    let by_mask = identify_unique_adjacency_masks(posterior, ctx, |mask, _inner| {
        (|| -> Result<Option<(IdentifiedEstimand, IdentificationResult)>, CausalError> {
            let Ok(admg) = admg_from_adjacency_mask(mask, posterior.n_vars()) else {
                return Ok(None);
            };
            let Some(identification) =
                class_envelope_outcome(identify_admg(DEFAULT_ADMG_IDENTIFIER_ID, &admg, query))?
            else {
                return Ok(None);
            };
            if !crate::analysis::execute::identification_status_ok_for_case(identification.status)
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
    collect_mask_atoms(posterior, ctx, &by_mask)
}

/// Identify every atom of a lagged temporal-ADMG posterior on the unrolled window of its own
/// template (the complete ID algorithm, as a supplied `TemporalAdmg` contrast is).
///
/// Atoms are keyed by posterior position: two atoms that differ only in their lagged edges
/// share a contemporaneous mask, so the posterior's graph keys do not tell them apart. An atom
/// whose template the window cannot hold, or whose contrast the window leaves unidentified (a
/// hedge, a modifier a treated period can reach), keeps its mass as unidentified mass.
///
/// # Errors
///
/// A posterior that is not a lagged temporal-ADMG posterior, a population other than the
/// observed one (a property of the query, not of an atom), or cancellation.
pub(crate) fn build_temporal_admg_posterior_identification_cache(
    posterior: &GraphPosterior,
    variables: &[antecedent_core::VariableId],
    query: &antecedent_core::TemporalEffectQuery,
    ctx: &ExecutionContext,
) -> Result<CachedGraphPosteriorIdentification, CausalError> {
    use crate::analysis::temporal_admg_contrast::identify_window_contrast;

    if !posterior.is_lagged_temporal_admg() {
        return Err(CausalError::Compile {
            message: "a lagged temporal-ADMG identification requires TemporalAdmg atoms with \
                      per-atom lag masks"
                .into(),
        });
    }
    if query.target_population != antecedent_core::TargetPopulation::AllObserved {
        return Err(CausalError::Compile {
            message: "a contrast over lagged temporal-ADMG atoms answers for the observed \
                      population; a reweighted or local target population is a different \
                      estimand"
                .into(),
        });
    }
    let max_lag = posterior.max_lag().unwrap_or(0);
    let variable_count = u32::try_from(variables.len()).map_err(|_| CausalError::Compile {
        message: "series schema exceeds the variable id range".into(),
    })?;
    crate::analysis::execute::report_identify_compute(ctx);
    let mapped = map_posterior_graphs(posterior, ctx, |i, _inner| {
        let Some(atom) = posterior.temporal_admg_atom(i) else { return Ok(None) };
        let Ok(template) = atom.to_template(posterior.n_vars(), max_lag, variables) else {
            return Ok(None);
        };
        let Ok(contrast) = identify_window_contrast(&template, query, variable_count) else {
            return Ok(None);
        };
        Ok::<Option<(IdentifiedEstimand, IdentificationResult)>, CausalError>(Some((
            contrast.estimand,
            contrast.identification,
        )))
    })?;
    let mut weights = Vec::with_capacity(posterior.n_graphs());
    let mut flags = Vec::with_capacity(posterior.n_graphs());
    let mut keys = Vec::with_capacity(posterior.n_graphs());
    let mut atoms = Vec::new();
    for (i, found) in mapped.into_iter().enumerate() {
        let key = super::dbn_envelope_key(i)?;
        keys.push(key);
        weights.push(posterior.weights()[i]);
        if let Some((estimand, identification)) = found {
            flags.push(GraphIdentFlag::Identified);
            atoms.push(CachedGraphPosteriorAtomIdentification {
                key,
                estimand,
                identification,
                transport: None,
            });
        } else {
            flags.push(GraphIdentFlag::Unidentified);
        }
    }
    Ok(CachedGraphPosteriorIdentification {
        graphs: WeightedGraphSamples::new(weights, flags, keys)?,
        atoms: Arc::from(atoms),
        class_atoms: Arc::from([]),
    })
}

/// Only an identification refusal can remove an atom from the identified mass.
fn class_envelope_outcome<T>(outcome: Result<T, CausalError>) -> Result<Option<T>, CausalError> {
    match outcome {
        Ok(envelope) => Ok(Some(envelope)),
        Err(error) if error.is_identification_refusal() => Ok(None),
        Err(error) => Err(error),
    }
}

/// Identify every CPDAG/PAG posterior atom with the existing class ATE envelope.
fn build_class_graph_posterior_identification_cache(
    posterior: &GraphPosterior,
    query: &AverageEffectQuery,
    ctx: &ExecutionContext,
) -> Result<CachedGraphPosteriorIdentification, CausalError> {
    use crate::strategy_table::{
        DEFAULT_PAG_IDENTIFIER_ID, identify_cpdag_in_context, identify_pag_in_context,
    };
    use antecedent_discovery::adjacency_mask::{
        cpdag_from_adjacency_mask, pag_from_adjacency_mask,
    };

    crate::analysis::execute::report_identify_compute(ctx);
    let resolved = map_posterior_graphs(posterior, ctx, |i, inner| {
        let mask = &posterior.atoms()[i];
        let mark = posterior.mark_atom(i);
        let key = posterior.graph_keys()[i];
        let cached = match posterior.atom_kind() {
            antecedent_discovery::GraphPosteriorAtomKind::Cpdag => {
                let Ok(cpdag) = cpdag_from_adjacency_mask(mask, posterior.n_vars()) else {
                    return Ok(None);
                };
                class_envelope_outcome(identify_cpdag_in_context(
                    DEFAULT_PAG_IDENTIFIER_ID,
                    &cpdag,
                    query,
                    Some(inner),
                ))
                .map(|envelope| envelope.map(|envelope| cache_class_envelope(key, query, envelope)))
            }
            antecedent_discovery::GraphPosteriorAtomKind::Pag => {
                let Ok(pag) = pag_from_adjacency_mask(mask, &mark, posterior.n_vars()) else {
                    return Ok(None);
                };
                class_envelope_outcome(identify_pag_in_context(
                    DEFAULT_PAG_IDENTIFIER_ID,
                    &pag,
                    query,
                    Some(inner),
                ))
                .map(|envelope| envelope.map(|envelope| cache_class_envelope(key, query, envelope)))
            }
            _ => Ok(None),
        }?;
        Ok(cached)
    })?;
    let mut resolved = resolved.into_iter();
    let (graphs, class_atoms) =
        collect_graph_posterior_atoms(posterior, ctx, |_, first_for_key| {
            let Some(cached) = resolved.next().flatten() else {
                return Ok((GraphIdentFlag::Unidentified, None));
            };
            // Envelope-level GraphDependent still carries identified completion mass.
            // Only atoms with zero identified completion weight are posterior-unidentified.
            let identified = cached.identified_weight > 0.0;
            let flag =
                if identified { GraphIdentFlag::Identified } else { GraphIdentFlag::Unidentified };
            Ok((flag, (first_for_key && identified).then_some(cached)))
        })?;
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

    let identification =
        crate::analysis::execute::envelope_to_identification_result(&envelope, query);
    let cases: Vec<CachedClassPosteriorCase> = envelope
        .cases
        .iter()
        .map(|case| {
            let estimand =
                if crate::analysis::execute::identification_status_ok_for_case(case.result.status)
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

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::VariableId;
    use antecedent_discovery::adjacency_mask::set_edge;
    use antecedent_discovery::{AdjacencyMask, SamplingSummary};
    use antecedent_prob::InferenceDiagnostics;

    /// A static posterior whose samples repeat a handful of confounding structures, as an
    /// MCMC posterior does: many entries, few distinct graphs.
    fn repeated_posterior(n_graphs: usize) -> GraphPosterior {
        const N: usize = 4;
        let edges = |list: &[(usize, usize)]| {
            list.iter().fold(AdjacencyMask::EMPTY, |mask, &(from, to)| {
                set_edge(mask, N, from, to, true)
            })
        };
        let structures = [
            edges(&[(0, 1)]),
            edges(&[(0, 1), (2, 0)]),
            edges(&[(0, 1), (2, 1)]),
            edges(&[(0, 1), (2, 0), (2, 1)]),
            edges(&[(0, 1), (2, 0), (3, 1)]),
            edges(&[(0, 1), (3, 0), (3, 1), (2, 3)]),
            edges(&[(2, 0), (2, 1)]),
            edges(&[(0, 1), (1, 2)]),
        ];
        let adjacency: Vec<AdjacencyMask> =
            (0..n_graphs).map(|i| structures[i % structures.len()].clone()).collect();
        GraphPosterior::from_atoms(
            N,
            vec![1.0 / n_graphs as f64; n_graphs],
            adjacency,
            vec![0.0; N * N],
            vec![0.0; N * N],
            SamplingSummary {
                ess: n_graphs as f64,
                diagnostics: InferenceDiagnostics::analytic("identification_cache_timing"),
                rejected_invalid: 0,
            },
        )
        .unwrap()
    }

    fn class_posterior() -> GraphPosterior {
        let posterior = repeated_posterior(2);
        let mut complete = AdjacencyMask::EMPTY;
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    complete = set_edge(complete, 4, i, j, true);
                }
            }
        }
        GraphPosterior::from_atoms(
            posterior.n_vars(),
            Arc::clone(posterior.weights()),
            vec![complete; 2],
            Arc::clone(posterior.edge_marginals()),
            Arc::clone(posterior.orientation_marginals()),
            SamplingSummary {
                ess: posterior.ess(),
                diagnostics: posterior.diagnostics().clone(),
                rejected_invalid: posterior.rejected_invalid(),
            },
        )
        .unwrap()
        .with_atom_kind(antecedent_discovery::GraphPosteriorAtomKind::Cpdag)
    }

    #[test]
    fn static_class_posterior_workers_keep_the_actual_pool_after_a_warm_run() {
        use antecedent_core::CachePolicy;
        use antecedent_core::execution::AdmissionPool;
        let posterior = class_posterior();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let warm = build_graph_posterior_identification_cache(
            &posterior,
            &query,
            &ExecutionContext::production(81, 2),
        )
        .unwrap();
        assert!(!warm.class_atoms.is_empty());
        for policy in
            [CachePolicy::disabled(), CachePolicy::enabled(Some(1)), CachePolicy::enabled(None)]
        {
            let pool = AdmissionPool::fixed(2, 1 << 20);
            let mut ctx = ExecutionContext::production(81, 2).with_admission(pool.clone());
            ctx.cache_policy = policy;
            let held = ctx.acquire_admission(0, (1 << 20) - 1).unwrap();
            let error =
                build_graph_posterior_identification_cache(&posterior, &query, &ctx).unwrap_err();
            assert!(matches!(error, CausalError::Resource { .. }), "{error:?}");
            assert_eq!(pool.available().bytes(), 1);
            drop(held);
            assert_eq!(pool.available().bytes(), 1 << 20);
        }
    }

    #[test]
    fn static_class_posterior_worker_cancellation_is_not_unidentified_mass() {
        let posterior = class_posterior();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let mut ctx = ExecutionContext::production(81, 2);
        ctx.cancellation = antecedent_core::CancellationToken::cancel_after_checks(32);
        let error =
            build_graph_posterior_identification_cache(&posterior, &query, &ctx).unwrap_err();
        assert!(matches!(error, CausalError::Cancelled { .. }), "{error:?}");
        assert_eq!(ctx.admission.available().threads(), 2);
    }

    /// Timing harness for the three static builders (`cargo test -p antecedent --lib
    /// identification_cache::graph_posterior::tests::builder_scales_with_samples -- --ignored --nocapture`).
    #[test]
    #[ignore = "timing harness"]
    fn builder_scales_with_samples() {
        let ctx = ExecutionContext::for_tests(11);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        for n_graphs in [2_000_usize, 20_000, 200_000] {
            let posterior = repeated_posterior(n_graphs);
            let started = std::time::Instant::now();
            let cache = build_graph_posterior_identification_cache(&posterior, &query, &ctx)
                .expect("identification cache");
            let elapsed = started.elapsed();
            eprintln!(
                "n_graphs={n_graphs:>7} atoms={} build={:.1} ms",
                cache.atoms.len(),
                elapsed.as_secs_f64() * 1e3
            );
        }
    }
}
