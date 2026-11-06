//! Mechanism attribution and simulation routes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Admg, Arc, AttributeQuery, AttributeTarget, AttributionOutcome, CELL_BYTES, CausalError,
    CausalRng, ChangeAttributionResult, ChangeEstimator, ChangeMeasure, Dag, DenseNodeId,
    DifferenceMeasure, DistributionChangeOptions, ExecutionContext, FitResampling, FittedGcm,
    InfluenceAllocation, IntrinsicAllocation, IntrinsicInfluenceQuery, MechanismChangeMethod,
    MechanismChangeTest, NoiseDeclaration, NoiseIndependence, RobustChangeOptions, SimulateQuery,
    SimulationOutcome, SimulationTarget, TabularData, ValueBatch, VariableId, history_batches,
    panel_tables,
};

/// Bytes a fitted-model route leases before it runs.
pub(crate) fn model_quote(target_rows: u64, nodes: u64, draws: u64) -> u64 {
    target_rows.max(draws).saturating_mul(nodes.max(1)).saturating_mul(CELL_BYTES).saturating_mul(3)
}

pub(crate) fn difference_measure(measure: ChangeMeasure) -> DifferenceMeasure {
    match measure {
        ChangeMeasure::MeanDifference => DifferenceMeasure::MeanDiff,
        ChangeMeasure::VarianceDifference => DifferenceMeasure::VarianceDiff,
        ChangeMeasure::GaussianKl => DifferenceMeasure::GaussianKl,
    }
}

fn change_attribution(
    fitted: &FittedGcm,
    data: &TabularData,
    change: &antecedent_core::ChangeAttributionQuery,
    estimator: &ChangeEstimator,
    ctx: &ExecutionContext,
) -> Result<ChangeAttributionResult, CausalError> {
    match estimator {
        ChangeEstimator::Sampled { measure, n_samples, seed } => {
            crate::gcm::attribute_distribution_change(
                &fitted.model,
                data,
                change,
                &DistributionChangeOptions {
                    measure: difference_measure(*measure),
                    n_samples: *n_samples,
                    seed: *seed,
                },
                ctx,
            )
        }
        ChangeEstimator::Robust { max_rows, n_samples, seed } => {
            crate::gcm::attribute_distribution_change_robust(
                &fitted.model,
                data,
                change,
                &RobustChangeOptions { max_rows: *max_rows, n_samples: *n_samples, seed: *seed },
                ctx,
            )
        }
    }
}

/// The structure an intrinsic-influence query declares the noise on: the
/// model's directed edges and the declared bidirected pairs.
fn noise_structure(
    graph: &Dag,
    bidirected: &[(VariableId, VariableId)],
) -> Result<Admg, CausalError> {
    let nodes = u32::try_from(graph.node_count()).map_err(|_| CausalError::Resource {
        message: "the model has more nodes than a node id holds".into(),
    })?;
    let mut admg = Admg::with_variables(nodes);
    for (from, to) in graph.directed_edge_pairs_sorted() {
        admg.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to))
            .map_err(CausalError::rejected)?;
    }
    for (a, b) in bidirected {
        admg.insert_bidirected(DenseNodeId::from_raw(a.raw()), DenseNodeId::from_raw(b.raw()))
            .map_err(CausalError::rejected)?;
    }
    Ok(admg)
}

/// How a mechanism-change detection draws its null law and reads its answer.
///
/// `block` is the block length of a dependent series (the null law is then a block permutation
/// of the residuals, not an independent-residual one); `bayesian` reads the answer as the
/// posterior probability of a change under the prior the query declares.
type SelectedRunPair<'a> = (&'a [std::ops::Range<usize>], &'a [std::ops::Range<usize>]);

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ChangeReading<'a> {
    /// Block length of a dependent series, when the populations are segments of one.
    pub(crate) block: Option<usize>,
    /// Read the answer as a posterior probability of a change.
    pub(crate) bayesian: bool,
    pub(crate) runs: Option<SelectedRunPair<'a>>,
    pub(crate) budget: Option<&'a antecedent_core::TemporalBudget>,
}

fn temporal_attribution_error(error: antecedent_attribution::AttributionError) -> CausalError {
    match error {
        antecedent_attribution::AttributionError::Model(
            error @ (antecedent_model::ModelError::Cancelled
            | antecedent_model::ModelError::ResourceLimit { .. }),
        ) => CausalError::from(error),
        other => CausalError::from(other),
    }
}

/// Block permutations a dependent-series mechanism-change test runs.
pub(crate) const CHANGE_BLOCK_PERMUTATIONS: usize = 499;

/// The detections of a mechanism-change query under `reading`: the posterior probability of a
/// change under the declared prior, the block-permutation test of a dependent series, or the
/// independent-residual test.
pub(crate) fn run_mechanism_change(
    fitted: &FittedGcm,
    data: &TabularData,
    detection: &antecedent_core::MechanismChangeQuery,
    method: MechanismChangeMethod,
    reading: ChangeReading<'_>,
    ctx: &ExecutionContext,
) -> Result<Vec<antecedent_attribution::MechanismChangeDetection>, CausalError> {
    if reading.bayesian {
        let prior = detection
            .change_prior
            .map(antecedent_core::OrderedFloatBits::to_f64)
            .ok_or_else(|| {
                crate::unsupported_reason!(
                    "option_not_applicable",
                    "the posterior probability of a mechanism change needs a prior over the change \
                 indicator, and none is assumed. Next attempt: declare it with \
                 MechanismChangeQuery::with_change_prior"
                )
            })?;
        let posterior = if let Some((baseline, comparison)) = reading.runs {
            let mut budget = reading.budget.cloned().ok_or_else(|| CausalError::Compile {
                message: "temporal mechanism detection lost its shared budget".into(),
            })?;
            antecedent_attribution::mechanism_change::mechanism_change_posterior_with_runs(
                &fitted.model,
                data,
                detection,
                prior,
                reading.block.unwrap_or(1),
                baseline,
                comparison,
                &mut budget,
                ctx,
            )
            .map_err(temporal_attribution_error)?
        } else {
            antecedent_attribution::mechanism_change_posterior(
                &fitted.model,
                data,
                detection,
                prior,
                reading.block.unwrap_or(1),
                ctx,
            )?
        };
        return Ok(posterior
            .into_iter()
            .map(antecedent_attribution::MechanismChangePosterior::into_detection)
            .collect());
    }
    if let Some(block) = reading.block {
        if let Some((baseline, comparison)) = reading.runs {
            let mut budget = reading.budget.cloned().ok_or_else(|| CausalError::Compile {
                message: "temporal mechanism detection lost its shared budget".into(),
            })?;
            return antecedent_attribution::mechanism_change::detect_mechanism_changes_block_with_runs(
                    &fitted.model,
                    data,
                    detection,
                    method,
                    block,
                    CHANGE_BLOCK_PERMUTATIONS,
                    baseline,
                    comparison,
                    &mut budget,
                    ctx,
                )
                .map_err(temporal_attribution_error);
        }
        return Ok(antecedent_attribution::detect_mechanism_changes_block(
            &fitted.model,
            data,
            detection,
            method,
            block,
            CHANGE_BLOCK_PERMUTATIONS,
            ctx,
        )?);
    }
    crate::gcm::mechanism_change_detection(&fitted.model, data, detection, method, ctx)
}

/// The mechanism-fit refits an attribution carries: how many and how they weight the rows.
///
/// The study's inference mode sets it (a posterior over the mechanisms under Bayesian
/// inference), so the query itself, whose identity is canonical, does not change with it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AttributionFit {
    /// Refits to run.
    pub(crate) replicates: u32,
    /// The row weighting of each refit.
    pub(crate) resampling: FitResampling,
}

/// The named scalar quantities of a change attribution: its total, one entry per component, and
/// one per path when it decomposes along paths.
pub(crate) fn change_quantities(result: &ChangeAttributionResult) -> Vec<(String, f64)> {
    let mut values = vec![("total_change".to_owned(), result.total_change)];
    values.extend(
        result
            .contributions
            .iter()
            .map(|c| (format!("component[{}]", c.component.variable().raw()), c.contribution)),
    );
    values.extend(result.path_breakdown.iter().map(|path| {
        let nodes: Vec<String> = path.path.iter().map(|v| v.raw().to_string()).collect();
        (format!("path[{}]", nodes.join(">")), path.contribution)
    }));
    values
}

/// The named scalar quantities of a simulation, all smooth functionals of the fitted mechanisms:
/// the mean of each variable of a draw batch (interventional draws, noise samples), the policy
/// and baseline means and their paired difference, the mean unit effect, the mean of each
/// outcome in each world of a counterfactual, and, for a rollout, the mean of each of the first
/// `nodes` variables at each step (step-major).
pub(crate) fn simulation_quantities(
    outcome: &SimulationOutcome,
    nodes: usize,
) -> Option<Vec<(String, f64)>> {
    // An empty column has no mean: summarising it is unavailable rather than a zero mean.
    let mean = |column: &[f64]| -> Option<f64> {
        (!column.is_empty()).then(|| antecedent_kernels::mean(column))
    };
    let batch_means = |batch: &ValueBatch| -> Option<Vec<(String, f64)>> {
        (0..batch.n_nodes)
            .map(|node| Some((format!("draw_mean[{node}]"), mean(batch.column(node).ok()?)?)))
            .collect()
    };
    match outcome {
        SimulationOutcome::Sample(batch) => batch_means(batch),
        SimulationOutcome::NoiseSample { draws, .. } => batch_means(draws),
        SimulationOutcome::PolicyValue(value) => Some(vec![
            ("policy_mean".to_owned(), value.policy_mean),
            ("baseline_mean".to_owned(), value.baseline_mean),
            ("mean_difference".to_owned(), value.mean_difference),
        ]),
        SimulationOutcome::UnitEffects(ite) => Some(vec![("mean_ite".to_owned(), ite.mean_ite)]),
        SimulationOutcome::Counterfactual(result) => {
            let mut out = Vec::new();
            for world in 0..result.n_worlds {
                for &outcome in result.outcomes.iter() {
                    let column = result.outcome_column(world, outcome).ok()?;
                    out.push((
                        format!("world[{world}].outcome[{}]", outcome.as_usize()),
                        mean(column)?,
                    ));
                }
            }
            Some(out)
        }
        SimulationOutcome::Temporal(path) => {
            let limit = nodes.min(path.n_nodes);
            let mut out = Vec::with_capacity(path.steps.len() * limit);
            for (step, batch) in path.steps.iter().enumerate() {
                for node in 0..limit {
                    out.push((
                        format!("step[{step}].node[{node}]"),
                        mean(batch.column(node).ok()?)?,
                    ));
                }
            }
            Some(out)
        }
    }
}

/// One simulated path of a rollout: the value of each of the first `nodes` variables at each step
/// (step-major, first row), innovations included. Rolled out under a fresh seed in every refit,
/// these are draws from the predictive law of a path: the refit uncertainty mixed with the
/// innovation noise.
pub(crate) fn simulated_path_quantities(
    outcome: &SimulationOutcome,
    nodes: usize,
) -> Option<Vec<(String, f64)>> {
    let SimulationOutcome::Temporal(path) = outcome else {
        return None;
    };
    let limit = nodes.min(path.n_nodes);
    let mut out = Vec::with_capacity(path.steps.len() * limit);
    for (step, batch) in path.steps.iter().enumerate() {
        for node in 0..limit {
            out.push((format!("step[{step}].path[{node}]"), *batch.column(node).ok()?.first()?));
        }
    }
    Some(out)
}

/// The named scalar quantities of an attribution under the row `weights` of a replicate (the
/// anomaly scores average under them), or `None` for an outcome without scalar quantities
/// (intrinsic influence returns its own refits; a detection is a test).
pub(crate) fn attribution_quantities_weighted(
    outcome: &AttributionOutcome,
    weights: &[f64],
) -> Option<Vec<(String, f64)>> {
    Some(match outcome {
        AttributionOutcome::Change(result) => change_quantities(result),
        AttributionOutcome::RootCauses { attribution, .. } => change_quantities(attribution),
        AttributionOutcome::FeatureRelevance(features) => {
            features.iter().map(|f| (format!("feature[{}]", f.feature.raw()), f.score)).collect()
        }
        AttributionOutcome::Anomaly(scores) => scores
            .iter()
            .map(|s| {
                let (mut numerator, mut denominator) = (0.0_f64, 0.0_f64);
                for (score, &row) in s.scores.iter().zip(s.rows.iter()) {
                    let weight = weights.get(row).copied().unwrap_or(0.0);
                    numerator += weight * score;
                    denominator += weight;
                }
                let mean = if denominator > 0.0 { numerator / denominator } else { f64::NAN };
                (format!("mean_anomaly_score[{}]", s.target.raw()), mean)
            })
            .collect(),
        AttributionOutcome::UnitChange(result) => result
            .components
            .iter()
            .zip(result.mean_contributions.iter())
            .map(|(component, mean)| (format!("unit_mean[{}]", component.raw()), *mean))
            .collect(),
        AttributionOutcome::IntrinsicInfluence(result) => result
            .sources
            .iter()
            .map(|source| (format!("source[{}]", source.source.raw()), source.contribution))
            .collect(),
        _ => return None,
    })
}

/// The named scalar quantities of an attribution at its point answer, in the order its posterior
/// and bootstrap columns use: a change's total, components and paths; feature scores; the mean
/// anomaly score per target; the mean unit contributions; each source's variance share. `None`
/// for a mechanism-change detection, which is a test and has no scalar quantities.
#[must_use]
pub fn attribution_quantities(outcome: &AttributionOutcome) -> Option<Vec<(String, f64)>> {
    let rows = match outcome {
        AttributionOutcome::Anomaly(scores) => {
            scores.iter().flat_map(|s| s.rows.iter().copied()).max().map_or(0, |max| max + 1)
        }
        _ => 0,
    };
    attribution_quantities_weighted(outcome, &vec![1.0; rows])
}

/// The compiled model of the comparison structure a structure-change attribution declares, over
/// the variables of `graph`.
///
/// # Errors
///
/// A model with more nodes than a node id holds, or a comparison edge the graph refuses.
pub(crate) fn structure_comparison_model(
    graph: &Dag,
    comparison: &[(VariableId, VariableId)],
) -> Result<crate::gcm::CompiledCausalModel, CausalError> {
    let nodes = u32::try_from(graph.node_count()).map_err(|_| CausalError::Resource {
        message: "the model has more nodes than a node id holds".into(),
    })?;
    let mut comparison_graph = Dag::with_variables(nodes);
    for (from, to) in comparison {
        comparison_graph
            .insert_directed(DenseNodeId::from_raw(from.raw()), DenseNodeId::from_raw(to.raw()))
            .map_err(CausalError::rejected)?;
    }
    Ok(crate::gcm::CompiledCausalModel::compile(comparison_graph)?)
}

/// Run the attribution `query` states on `fitted`, with the engine every `attribute_*` entry
/// runs. The mechanism-fit refits of an intrinsic-influence target are the query's own unless
/// `fit` sets them.
#[allow(clippy::too_many_lines, reason = "one exhaustive match over every attribution target")]
#[allow(
    clippy::too_many_arguments,
    reason = "the fitted attribution operation shares its graph, data and execution context"
)]
pub(crate) fn run_attribute_with_fit(
    fitted: &FittedGcm,
    graph: &Dag,
    data: &TabularData,
    query: &AttributeQuery,
    fit: Option<AttributionFit>,
    reading: ChangeReading<'_>,
    temporal_runs: Option<&[std::ops::Range<usize>]>,
    temporal_budget: Option<&antecedent_core::TemporalBudget>,
    ctx: &ExecutionContext,
) -> Result<AttributionOutcome, CausalError> {
    Ok(match &query.target {
        AttributeTarget::IntrinsicInfluence(target) => {
            let structure = match &target.noise {
                NoiseDeclaration::Structure { bidirected } => {
                    Some(noise_structure(graph, bidirected)?)
                }
                NoiseDeclaration::Undeclared | NoiseDeclaration::Independent => None,
            };
            let noise = match (&structure, &target.noise) {
                (Some(admg), _) => NoiseIndependence::Structure(admg),
                (None, NoiseDeclaration::Independent) => NoiseIndependence::Declared,
                (None, _) => NoiseIndependence::Undeclared,
            };
            let allocation = match target.allocation {
                InfluenceAllocation::Auto => IntrinsicAllocation::Auto,
                InfluenceAllocation::Exact => IntrinsicAllocation::Exact,
                InfluenceAllocation::Permutation { n_permutations } => {
                    IntrinsicAllocation::Permutation { n_permutations }
                }
            };
            let engine_query = IntrinsicInfluenceQuery::new(target.target)
                .with_allocation(allocation)
                .with_simulation(target.outer_draws, target.inner_draws, target.replicates)
                .with_fit_bootstrap(fit.map_or(target.fit_bootstrap_replicates, |f| f.replicates))
                .with_fit_resampling(fit.map_or(FitResampling::RowBootstrap, |f| f.resampling))
                .with_level(target.level)
                .with_max_sources(target.max_sources)
                .with_seed(target.seed);
            AttributionOutcome::IntrinsicInfluence(
                antecedent_attribution::intrinsic_influence::intrinsic_influence_with_temporal_runs(
                    &fitted.model, noise, Some(data), &engine_query, temporal_runs, temporal_budget, ctx,
                ).map_err(CausalError::from)?,
            )
        }
        AttributeTarget::RootCauses { change, estimator } => {
            let attribution = change_attribution(fitted, data, change, estimator, ctx)?;
            let ranks = crate::gcm::rank_root_causes(&attribution, ctx)?;
            AttributionOutcome::RootCauses { attribution, ranks }
        }
        AttributeTarget::DistributionChange { change, estimator } => {
            AttributionOutcome::Change(change_attribution(fitted, data, change, estimator, ctx)?)
        }
        AttributeTarget::Paths { sources, outcome, max_paths, max_len, through } => {
            let mut result = crate::gcm::attribute_paths(
                &fitted.model,
                sources,
                *outcome,
                *max_paths,
                *max_len,
                ctx,
            )?;
            if !through.is_empty() {
                // The same filter and total `attribute_path_specific` applies.
                let filtered: Vec<_> = result
                    .path_breakdown
                    .iter()
                    .filter(|path| through.iter().all(|node| path.path.iter().any(|v| v == node)))
                    .cloned()
                    .collect();
                result.total_change = filtered.iter().map(|path| path.contribution).sum();
                result.path_breakdown = Arc::from(filtered);
            }
            AttributionOutcome::Change(result)
        }
        AttributeTarget::FeatureRelevance { outcome, features, delta, n_samples, max_features } => {
            AttributionOutcome::FeatureRelevance(crate::gcm::attribute_feature_relevance_with(
                &fitted.model,
                data,
                *outcome,
                features,
                &antecedent_attribution::FeatureRelevanceOptions {
                    delta: *delta,
                    n_samples: *n_samples,
                    max_features: *max_features,
                },
                ctx,
            )?)
        }
        AttributeTarget::Anomaly(anomaly) => AttributionOutcome::Anomaly(
            crate::gcm::anomaly_attribution_query_with(&fitted.model, data, anomaly, ctx)?,
        ),
        AttributeTarget::UnitChange(unit) => AttributionOutcome::UnitChange(
            crate::gcm::attribute_unit_change(&fitted.model, data, unit, ctx)?,
        ),
        AttributeTarget::StructureChange { change, comparison, measure, n_samples, seed } => {
            let comparison_model = structure_comparison_model(graph, comparison)?;
            AttributionOutcome::Change(crate::gcm::attribute_structure_change(
                &fitted.model,
                &comparison_model,
                data,
                change,
                &crate::gcm::StructureChangeOptions {
                    measure: difference_measure(*measure),
                    n_samples: *n_samples,
                    seed: *seed,
                },
                ctx,
            )?)
        }
        AttributeTarget::MechanismChange { query: detection, test } => {
            let method = match test {
                MechanismChangeTest::LikelihoodRatio => MechanismChangeMethod::LikelihoodRatio,
                MechanismChangeTest::MeanDifference => MechanismChangeMethod::MeanDiff,
                MechanismChangeTest::ClassifierTwoSample => {
                    MechanismChangeMethod::ClassifierTwoSample
                }
                MechanismChangeTest::KernelTwoSample => MechanismChangeMethod::KernelTwoSample,
                MechanismChangeTest::ChangePoint => MechanismChangeMethod::ChangePoint,
            };
            AttributionOutcome::MechanismChange(run_mechanism_change(
                fitted, data, detection, method, reading, ctx,
            )?)
        }
    })
}

/// Run the simulation `query` states on `fitted`, with the engine every
/// `sample_do`, `policy_value` and `counterfactual_ite` entry runs.
pub(crate) fn run_simulate(
    fitted: &FittedGcm,
    data: &TabularData,
    query: &SimulateQuery,
    ctx: &ExecutionContext,
) -> Result<SimulationOutcome, CausalError> {
    Ok(match &query.target {
        SimulationTarget::InterventionalSample { interventions, draws, seed } => {
            let mut rng = CausalRng::from_seed(*seed);
            SimulationOutcome::Sample(crate::gcm::sample_do(
                &fitted.model,
                interventions,
                *draws,
                &mut rng,
                ctx,
            )?)
        }
        SimulationTarget::PolicyValue { interventions, outcome, draws } => {
            SimulationOutcome::PolicyValue(crate::gcm::policy_value(
                &fitted.model,
                interventions,
                *outcome,
                *draws,
                ctx,
            )?)
        }
        SimulationTarget::UnitEffects { treatment, outcome, active, control } => {
            SimulationOutcome::UnitEffects(crate::gcm::counterfactual_ite(
                fitted.model.clone(),
                data,
                *treatment,
                *outcome,
                *active,
                *control,
                ctx,
            )?)
        }
        SimulationTarget::NoiseSample { interventions, draws } => {
            let (draws, noise) =
                crate::gcm::sample_with_noise(&fitted.model, interventions, *draws, ctx)?;
            SimulationOutcome::NoiseSample { draws, noise }
        }
        SimulationTarget::NoiseReplay { interventions, noise, n_rows, policy_salt } => {
            let table = crate::gcm::noise_table_from_columns(&fitted.model, *n_rows, noise, ctx)?;
            SimulationOutcome::Sample(crate::gcm::replay_noise(
                &fitted.model,
                interventions,
                &table,
                *policy_salt,
                ctx,
            )?)
        }
        SimulationTarget::Counterfactual { interventions, outcomes } => {
            SimulationOutcome::Counterfactual(crate::gcm::counterfactual_policy(
                fitted.model.clone(),
                data,
                interventions,
                outcomes,
                ctx,
            )?)
        }
        SimulationTarget::TemporalRollout { schedule, n_rows, n_steps, window, history, seed } => {
            let initial = history_batches(fitted, history, *n_rows)?;
            let plan = crate::gcm::TemporalPlan {
                sequence: schedule,
                n_steps: *n_steps,
                n_rows: *n_rows,
                window: *window,
                initial_history: &initial,
            };
            let mut rng = CausalRng::from_seed(*seed);
            SimulationOutcome::Temporal(crate::gcm::sample_temporal_policy(
                &fitted.model,
                &plan,
                &mut rng,
                ctx,
            )?)
        }
        SimulationTarget::TemporalCounterfactual {
            schedule,
            n_rows,
            n_steps,
            window,
            panel,
            history,
        } => {
            let initial = history_batches(fitted, history, *n_rows)?;
            let tables = panel_tables(data, panel, *n_rows, *n_steps)?;
            let plan = crate::gcm::TemporalPlan {
                sequence: schedule,
                n_steps: *n_steps,
                n_rows: *n_rows,
                window: *window,
                initial_history: &initial,
            };
            SimulationOutcome::Temporal(crate::gcm::counterfactual_temporal_policy(
                fitted.model.clone(),
                &tables,
                &plan,
                ctx,
            )?)
        }
    })
}

/// Rows a simulation draws or replays, for its admission quote.
pub(crate) fn simulated_rows(target: &SimulationTarget, data_rows: u64) -> u64 {
    match target {
        SimulationTarget::InterventionalSample { draws, .. }
        | SimulationTarget::PolicyValue { draws, .. }
        | SimulationTarget::NoiseSample { draws, .. } => *draws as u64,
        SimulationTarget::NoiseReplay { n_rows, .. } => *n_rows as u64,
        SimulationTarget::UnitEffects { .. } | SimulationTarget::Counterfactual { .. } => data_rows,
        SimulationTarget::TemporalRollout { n_rows, n_steps, .. }
        | SimulationTarget::TemporalCounterfactual { n_rows, n_steps, .. } => {
            (*n_rows as u64).saturating_mul(*n_steps as u64)
        }
    }
}
