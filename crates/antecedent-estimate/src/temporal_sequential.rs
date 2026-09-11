//! Linear sequential g-computation on an identified unfolded DAG.
//!
//! Multi-step Sustained and multi-step / joint `Sequence` overlays share this
//! engine. Sustained is the active-minus-control contrast over a window;
//! Sequence evaluates the interventional level under per-node Set / Soft
//! constant / Soft shift overlays. No second identifier.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

// Dense ids originate as u32; nonnegative row indices and bounded replicate counts are intentional.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, ExecutionContext, IdentificationStatus, Lag, TargetPopulation,
    TemporalEffectQuery, TemporalNodeKey, TemporalPolicy, VariableId,
};
use antecedent_data::{LaggedColumn, LaggedSampleWorkspace, TemporalIndexer, TimeSeriesData};
use antecedent_expr::{EstimandMethod, IdentifiedEstimand};
use antecedent_graph::{DenseNodeId, TemporalDag};
use antecedent_prob::{BayesLikelihood, PosteriorDraws, PosteriorQuantityKind, PosteriorSchema};
use antecedent_stats::{CompiledDesign, DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};

use crate::{
    BayesianGCompWorkspace, BayesianGComputationAte, CausalPosterior, EffectEstimate,
    EstimationError, OverlapPolicy, PreparedBayesianProblem,
};

/// One intervened unfolded node and the licensed overlay applied there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SequentialNodeOverlay {
    /// Template variable.
    pub variable: VariableId,
    /// Absolute time offset of the intervened copy.
    pub offset: i32,
    /// Hard `Set` / Soft `constant` level. `None` means an additive shift.
    pub level: Option<f64>,
    /// Additive shift applied to the factual mean when [`Self::level`] is `None`.
    pub shift: f64,
}

impl SequentialNodeOverlay {
    /// Assigned value under a linear additive mechanism.
    #[must_use]
    pub fn assigned(self, factual_mean: f64) -> f64 {
        self.level.unwrap_or(factual_mean + self.shift)
    }
}

/// What the sequential engine returns for one outcome node.
#[derive(Clone, Copy, Debug)]
enum SequentialEval {
    /// Active-minus-control contrast; every intervened node is overwritten by `delta`.
    Contrast { delta: f64 },
    /// Interventional outcome level under the per-node overlays.
    Level,
}

/// Fit every non-intervened ancestor's linear mechanism in the identified
/// unfolded DAG. Propagate the active-minus-control difference in topological
/// order, overwriting **every** treatment-time node in the sustained window.
/// Frequentist uncertainty uses a shared moving-block row bootstrap across all
/// equations. Bayesian uncertainty uses independent Gaussian priors across
/// stationary mechanisms, sharing each coefficient draw across its time copies.
pub fn estimate_sustained_window(
    data: &TimeSeriesData,
    graph: &TemporalDag,
    indexer: &TemporalIndexer,
    estimand: &IdentifiedEstimand,
    query: &TemporalEffectQuery,
    status: IdentificationStatus,
    assumptions: AssumptionSet,
    bootstrap_replicates: u32,
    bayesian: Option<&BayesianGComputationAte>,
    ctx: &ExecutionContext,
) -> Result<(EffectEstimate, Option<CausalPosterior>), EstimationError> {
    query.validate()?;
    let TemporalPolicy::Sustained { from, until } = query.policy else {
        return Err(EstimationError::unsupported(
            "sequential estimator requires a sustained window",
        ));
    };
    if from >= until
        || query.target_population != TargetPopulation::AllObserved
        || estimand.method_kind().ok() != Some(EstimandMethod::TemporalBackdoorUnfolded)
        || !matches!(
            status,
            IdentificationStatus::NonparametricallyIdentified
                | IdentificationStatus::IdentifiedUnderParametricRestrictions
        )
    {
        return Err(EstimationError::unsupported(
            "sequential estimator requires an identified unfolded multi-step contrast and AllObserved",
        ));
    }
    if bayesian.is_some_and(|est| est.likelihood != BayesLikelihood::GaussianIdentity) {
        return Err(EstimationError::unsupported(
            "sequential Bayesian g-computation requires GaussianIdentity",
        ));
    }
    if bayesian.is_some_and(|est| est.backend == crate::BayesianBackendKind::Hmc) {
        return Err(EstimationError::unsupported(
            "composed sustained HMC requires derived-contrast chain diagnostics; use conjugate or Laplace",
        ));
    }
    let active = crate::adjustment::intervention_f64(&query.active)?;
    let delta = active - crate::adjustment::intervention_f64(&query.control)?;
    let overlays: Vec<SequentialNodeOverlay> = (from..=until)
        .map(|offset| SequentialNodeOverlay {
            variable: query.treatment,
            offset,
            level: Some(active),
            shift: 0.0,
        })
        .collect();
    estimate_sequential(
        data,
        graph,
        indexer,
        estimand,
        query.outcome,
        query.outcome_offset(),
        &overlays,
        SequentialEval::Contrast { delta },
        status,
        assumptions,
        bootstrap_replicates,
        bayesian,
        ctx,
    )
}

/// Sequential g-computation of the interventional **level** under per-node
/// Set / Soft constant / Soft shift overlays. Shares the Sustained engine.
///
/// # Errors
///
/// Empty or duplicate overlays, unidentified estimand, or fit failures.
pub fn estimate_sequence_overlays(
    data: &TimeSeriesData,
    graph: &TemporalDag,
    indexer: &TemporalIndexer,
    estimand: &IdentifiedEstimand,
    outcome: VariableId,
    outcome_offset: i32,
    overlays: &[SequentialNodeOverlay],
    status: IdentificationStatus,
    assumptions: AssumptionSet,
    bootstrap_replicates: u32,
    bayesian: Option<&BayesianGComputationAte>,
    ctx: &ExecutionContext,
) -> Result<(EffectEstimate, Option<CausalPosterior>), EstimationError> {
    if overlays.is_empty() {
        return Err(EstimationError::unsupported("sequential overlay requires at least one step"));
    }
    if estimand.method_kind().ok() != Some(EstimandMethod::TemporalBackdoorUnfolded)
        || !matches!(
            status,
            IdentificationStatus::NonparametricallyIdentified
                | IdentificationStatus::IdentifiedUnderParametricRestrictions
        )
    {
        return Err(EstimationError::unsupported(
            "sequential overlay requires an identified unfolded design",
        ));
    }
    if bayesian.is_some_and(|est| est.likelihood != BayesLikelihood::GaussianIdentity) {
        return Err(EstimationError::unsupported(
            "sequential Bayesian g-computation requires GaussianIdentity",
        ));
    }
    if bayesian.is_some_and(|est| est.backend == crate::BayesianBackendKind::Hmc) {
        return Err(EstimationError::unsupported(
            "composed sequence HMC requires derived-contrast chain diagnostics; use conjugate or Laplace",
        ));
    }
    estimate_sequential(
        data,
        graph,
        indexer,
        estimand,
        outcome,
        outcome_offset,
        overlays,
        SequentialEval::Level,
        status,
        assumptions,
        bootstrap_replicates,
        bayesian,
        ctx,
    )
}

fn estimate_sequential(
    data: &TimeSeriesData,
    graph: &TemporalDag,
    indexer: &TemporalIndexer,
    estimand: &IdentifiedEstimand,
    outcome: VariableId,
    outcome_offset: i32,
    overlays: &[SequentialNodeOverlay],
    eval: SequentialEval,
    status: IdentificationStatus,
    assumptions: AssumptionSet,
    bootstrap_replicates: u32,
    bayesian: Option<&BayesianGComputationAte>,
    ctx: &ExecutionContext,
) -> Result<(EffectEstimate, Option<CausalPosterior>), EstimationError> {
    let unfolded =
        graph.unfold(indexer.clone()).map_err(|e| EstimationError::data_msg(e.to_string()))?;
    let dag = &unfolded.dag;
    let outcome = indexer
        .dense_id(TemporalNodeKey { variable: outcome, offset: outcome_offset })
        .map_err(|e| EstimationError::data_msg(e.to_string()))? as usize;
    let mut intervention = vec![false; dag.node_count()];
    let mut overlay_at = vec![None; dag.node_count()];
    for &overlay in overlays {
        let dense = indexer
            .dense_id(TemporalNodeKey { variable: overlay.variable, offset: overlay.offset })
            .map_err(|e| EstimationError::data_msg(e.to_string()))? as usize;
        if intervention[dense] {
            return Err(EstimationError::unsupported(
                "Sequence assigns the same (variable, time) twice; refuse rather than collapse",
            ));
        }
        intervention[dense] = true;
        overlay_at[dense] = Some(overlay);
    }
    let mut needed = vec![false; dag.node_count()];
    let mut pending = vec![outcome];
    while let Some(i) = pending.pop() {
        if needed[i] {
            continue;
        }
        needed[i] = true;
        if !intervention[i] {
            pending
                .extend(dag.parents(DenseNodeId::from_raw(i as u32)).iter().map(|p| p.as_usize()));
        }
    }
    let order: Vec<_> = dag
        .topological_order()
        .ok_or_else(|| EstimationError::data_msg("unfolded graph is not acyclic"))?
        .into_iter()
        .map(DenseNodeId::as_usize)
        .filter(|&i| needed[i])
        .collect();
    let anchor = order
        .iter()
        .map(|&i| indexer.key_of(i as u32).map(|k| k.offset))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| EstimationError::data_msg(e.to_string()))?
        .into_iter()
        .max()
        .unwrap_or(0)
        .max(0);
    let mut columns = Vec::new();
    let mut column_of = vec![0; dag.node_count()];
    let mut max_lag = 0;
    for &i in &order {
        let key = indexer.key_of(i as u32).map_err(|e| EstimationError::data_msg(e.to_string()))?;
        let lag = u32::try_from(anchor - key.offset)
            .map_err(|_| EstimationError::unsupported("invalid unfolded time offset"))?;
        max_lag = max_lag.max(lag);
        column_of[i] = columns.len();
        columns.push(LaggedColumn { variable: key.variable, lag: Lag::from_raw(lag) });
    }
    let plan = data.plan_lagged_sample(max_lag, Arc::from(columns))?;
    let mut workspace = LaggedSampleWorkspace::default();
    let sample = plan.prepare(data, &mut workspace, &ctx.kernel_policy)?;
    let n = sample.n;
    if n < 3 {
        return Err(EstimationError::unsupported(
            "sustained window requires at least three complete aligned rows",
        ));
    }
    let mut designs = vec![None; dag.node_count()];
    let mut parents = vec![Vec::new(); dag.node_count()];
    for &i in &order {
        parents[i] =
            dag.parents(DenseNodeId::from_raw(i as u32)).iter().map(|p| p.as_usize()).collect();
        // Stable coefficient positions across time copies, independent of dense
        // node order in the unfolding.
        let child = indexer.key_of(i as u32).expect("unfolded node");
        parents[i].sort_by_key(|&p| {
            let parent = indexer.key_of(p as u32).expect("unfolded parent");
            (parent.variable.raw(), child.offset - parent.offset)
        });
        if intervention[i] || parents[i].is_empty() {
            continue;
        }
        let t = sample.column(column_of[parents[i][0]]);
        let covs: Vec<_> = parents[i]
            .iter()
            .skip(1)
            .map(|&p| (VariableId::from_raw(p as u32), sample.column(column_of[p])))
            .collect();
        designs[i] =
            Some(CompiledDesign::linear_adjustment(t, &covs, sample.column(column_of[i]), &[])?);
    }
    let mut factual = vec![0.0; dag.node_count()];
    for &i in &order {
        let col = sample.column(column_of[i]);
        factual[i] = col.iter().sum::<f64>() / n as f64;
    }
    let propagate = |coefficients: &[Vec<f64>]| match eval {
        SequentialEval::Contrast { delta } => {
            let mut differences = vec![0.0; dag.node_count()];
            for &i in &order {
                differences[i] = if intervention[i] {
                    delta
                } else {
                    parents[i]
                        .iter()
                        .enumerate()
                        .map(|(p, &node)| coefficients[i][p + 1] * differences[node])
                        .sum()
                };
            }
            differences[outcome]
        }
        SequentialEval::Level => {
            let mut values = vec![0.0; dag.node_count()];
            for &i in &order {
                values[i] = if let Some(overlay) = overlay_at[i] {
                    overlay.assigned(factual[i])
                } else if coefficients[i].is_empty() {
                    factual[i]
                } else {
                    coefficients[i][0]
                        + parents[i]
                            .iter()
                            .enumerate()
                            .map(|(p, &node)| coefficients[i][p + 1] * values[node])
                            .sum::<f64>()
                };
            }
            values[outcome]
        }
    };
    let mut ls_ws = LeastSquaresWorkspace::default();
    let mut fit_ols = |rows: Option<&[usize]>| -> Result<Vec<Vec<f64>>, EstimationError> {
        let mut coefficients = vec![Vec::new(); dag.node_count()];
        for &i in &order {
            if let Some(design) = &designs[i] {
                let (matrix, y) = if let Some(rows) = rows {
                    (
                        design
                            .matrix
                            .chunks(n)
                            .flat_map(|column| rows.iter().map(|&r| column[r]))
                            .collect::<Vec<_>>(),
                        rows.iter().map(|&r| design.outcome[r]).collect::<Vec<_>>(),
                    )
                } else {
                    (design.matrix.to_vec(), design.outcome.to_vec())
                };
                coefficients[i] = FaerBackend
                    .least_squares(&matrix, y.len(), design.ncols, &y, &mut ls_ws)?
                    .coefficients;
            }
        }
        Ok(coefficients)
    };
    if let Some(estimator) = bayesian {
        let mut mechanism_posts = vec![None; dag.node_count()];
        let mut mechanism_of = vec![0; dag.node_count()];
        let mut mechanisms: Vec<(VariableId, Vec<LaggedColumn>, usize)> = Vec::new();
        let mut count = usize::MAX;
        for &i in &order {
            if designs[i].is_some() {
                let child = indexer.key_of(i as u32).expect("unfolded node");
                let parent_columns: Vec<_> = parents[i]
                    .iter()
                    .map(|&p| {
                        let parent = indexer.key_of(p as u32).expect("unfolded parent");
                        LaggedColumn {
                            variable: parent.variable,
                            lag: Lag::from_raw(
                                u32::try_from(child.offset - parent.offset).expect("causal lag"),
                            ),
                        }
                    })
                    .collect();
                if let Some((_, columns, owner)) =
                    mechanisms.iter().find(|(v, _, _)| *v == child.variable)
                {
                    if columns != &parent_columns {
                        return Err(EstimationError::unsupported(
                            "stationary mechanism has inconsistent unfolded parents",
                        ));
                    }
                    mechanism_of[i] = *owner;
                    continue;
                }
                // Fit each stationary mechanism once on its unique observed
                // time-series rows. Overlapping unfolded windows must not
                // multiply the likelihood or create independent copies of beta.
                let mut columns = parent_columns.clone();
                columns.push(LaggedColumn { variable: child.variable, lag: Lag::CONTEMPORANEOUS });
                let max_lag = columns.iter().map(|c| c.lag.raw()).max().unwrap_or(0);
                let plan = data.plan_lagged_sample(max_lag, Arc::from(columns))?;
                let mut workspace = LaggedSampleWorkspace::default();
                let sample = plan.prepare(data, &mut workspace, &ctx.kernel_policy)?;
                let covs: Vec<_> = parent_columns
                    .iter()
                    .enumerate()
                    .skip(1)
                    .map(|(j, c)| (c.variable, sample.column(j)))
                    .collect();
                let design = CompiledDesign::linear_adjustment(
                    sample.column(0),
                    &covs,
                    sample.column(parent_columns.len()),
                    &[],
                )?;
                let prep = PreparedBayesianProblem {
                    design,
                    method: estimand.method.clone(),
                    adjustment_set: Arc::from([]),
                    active: 1.0,
                    control: 0.0,
                    overlap: OverlapPolicy::ExplicitOverride,
                    coef_names: None,
                    unit_ids: None,
                };
                let mut est = estimator.clone();
                est.seed = est.seed.wrapping_add(u64::from(child.variable.raw()));
                let post = est.fit(&prep, status, &mut BayesianGCompWorkspace::default(), ctx)?;
                count = count.min(post.draws.n_draws);
                mechanism_posts[i] = Some(post);
                mechanism_of[i] = i;
                mechanisms.push((child.variable, parent_columns, i));
            }
        }
        let mut posterior = mechanism_posts.iter().flatten().next().cloned().ok_or_else(|| {
            EstimationError::unsupported("sustained contrast has no fitted outcome mechanism")
        })?;
        let mut values = Vec::with_capacity(count);
        for draw in 0..count {
            let mut coefficients = vec![Vec::new(); dag.node_count()];
            for &i in &order {
                if designs[i].is_some() {
                    let post = mechanism_posts[mechanism_of[i]].as_ref().expect("stationary fit");
                    for index in 0..designs[i].as_ref().expect("fitted design").ncols {
                        let col = post.draws.schema.quantities.iter().position(|q| matches!(q, PosteriorQuantityKind::Coefficient { index: j, .. } if *j == index)).ok_or_else(|| EstimationError::stats_msg("missing sequential coefficient"))?;
                        coefficients[i].push(post.draws.column(col)?[draw]);
                    }
                }
            }
            values.push(propagate(&coefficients));
        }
        let draws = PosteriorDraws::from_column_major(
            PosteriorSchema {
                quantities: Arc::from([PosteriorQuantityKind::Effect {
                    name: Arc::from("sustained_window"),
                }]),
            },
            count,
            values,
        )?;
        posterior.summaries = draws.summarize();
        posterior.draws = draws;
        for post in mechanism_posts.iter().flatten().skip(1) {
            posterior.assumptions.entries.extend(post.assumptions.entries.iter().cloned());
        }
        posterior.assumptions.entries.extend(assumptions.entries);
        let effect = EffectEstimate::new(
            posterior.summaries.mean[0],
            posterior.summaries.sd[0],
            posterior.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        return Ok((effect, Some(posterior)));
    }
    let point = propagate(&fit_ols(None)?);
    let mut draws = Vec::new();
    let mut failed = 0;
    // Blocks are at least the unfolded span, and grow with sample size.
    let block = (max_lag as usize + 1).max((n as f64).cbrt().ceil() as usize).min(n);
    for replicate in 0..bootstrap_replicates {
        if ctx.cancellation.is_cancelled() {
            break;
        }
        let mut rng = ctx.rng.stream(0x5120_0000 + u64::from(replicate));
        let mut rows = Vec::with_capacity(n);
        while rows.len() < n {
            let start = (rng.next_f64() * n as f64) as usize;
            for offset in 0..block {
                if rows.len() < n {
                    rows.push((start + offset) % n);
                }
            }
        }
        match fit_ols(Some(&rows)) {
            Ok(coefs) => draws.push(propagate(&coefs)),
            Err(_) => failed += 1,
        }
    }
    let se = if draws.len() > 1 {
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        Some(
            (draws.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (draws.len() - 1) as f64)
                .sqrt(),
        )
    } else {
        None
    };
    Ok((
        EffectEstimate::from_parts(
            point,
            f64::NAN,
            se,
            (bootstrap_replicates > 0).then_some(draws.len() as u32),
            (bootstrap_replicates > 0).then_some(failed),
            ctx.cancellation.is_cancelled(),
            false,
            assumptions,
            OverlapPolicy::ExplicitOverride,
            None,
            None,
        ),
        None,
    ))
}
