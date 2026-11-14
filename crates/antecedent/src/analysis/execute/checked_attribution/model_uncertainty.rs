//! Fitted-model attribution resampling and uncertainty.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    AttributionProcedure, CausalError, CausalPosterior, CausalQuery, CheckedAttributionOperation,
    Diagnostic, EffectEstimate, ExecutionContext, Replicates, SampledWeights, TableView,
    TabularData, admit_posterior_columns, attribute_intrinsic_target,
    attribution_quantities_weighted, change_quantities, difference_measure,
    disclose_model_bootstrap, intrinsic_posterior_columns, population_refit,
};

/// The replicate quantity a fitted-model query publishes as its point when that is not the first
/// quantity of its replicates: a policy value's point is the paired mean difference, listed
/// after the policy and baseline means.
fn published_quantity(query: &CausalQuery) -> Option<&'static str> {
    match query {
        CausalQuery::Simulate(simulate)
            if matches!(simulate.target, antecedent_core::SimulationTarget::PolicyValue { .. }) =>
        {
            Some("mean_difference")
        }
        _ => None,
    }
}

/// Posterior draws the coverage measurements of the fitted-model posteriors ran with.
const MEASURED_POSTERIOR_DRAWS: usize = 1_000;
/// Refit replicates the coverage measurements of the fitted-model bootstraps ran with.
const MEASURED_BOOTSTRAP_REPLICATES: u32 = 200;

/// Whether a replicate column is one simulated path of a rollout: a draw of the predictive law,
/// innovations included, rather than a refit of a mean.
fn mechanism_refit_path_column(name: &str) -> bool {
    name.contains("].path[")
}

/// Refer the refit draws of each mean-type column to the finite-sample posterior of a mean of
/// `units` exchangeable units in each of `populations` independent populations.
///
/// Under `Exp(1)` (Dirichlet) weights on `K` units a weighted mean has posterior variance
/// `s^2 (K - 1) / (K (K + 1))` and normal tails, where the reference posterior of a mean with an
/// unknown scale is `mean + s / sqrt(K) * t_{K-1}`. Each draw's deviation from the column mean
/// is therefore multiplied by `sqrt((K + 1) / (K - 1)) * sqrt(nu / chi2_nu)` with
/// `nu = populations * (K - 1)`: the first factor restores the variance, the second is the
/// scale mixture that turns normal tails into Student-t ones. One chi-square variate is shared
/// by every column of a draw and each column keeps its mean, so sums across columns (a total and
/// its components) are kept.
/// Columns that are simulated paths, or hold a non-finite value, are left as drawn.
#[allow(
    clippy::cast_precision_loss,
    reason = "unit and draw counts are far below the exact range of f64"
)]
pub(super) fn mechanism_refit_reference(
    columns: &mut [(String, Vec<f64>)],
    units: usize,
    populations: usize,
    rng: &mut antecedent_core::CausalRng,
) {
    let draws = columns.first().map_or(0, |(_, values)| values.len());
    if units < 3 || populations == 0 || draws == 0 {
        return;
    }
    let k = units as f64;
    let nu = populations as f64 * (k - 1.0);
    let restore = ((k + 1.0) / (k - 1.0)).sqrt();
    let scales: Vec<f64> = (0..draws)
        .map(|_| {
            // chi2_nu is Gamma(nu / 2, rate 1 / 2).
            let chi_square = antecedent_kernels::sample_gamma(0.5 * nu, 0.5, rng);
            restore * (nu / chi_square).sqrt()
        })
        .collect();
    if scales.iter().any(|scale| !scale.is_finite()) {
        return;
    }
    for (name, values) in columns.iter_mut() {
        if mechanism_refit_path_column(name)
            || values.len() != draws
            || values.iter().any(|value| !value.is_finite())
        {
            continue;
        }
        let centre = values.iter().sum::<f64>() / draws as f64;
        for (value, scale) in values.iter_mut().zip(&scales) {
            *value = (*value - centre) * scale;
        }
        // The scaled deviations are re-centred, so the posterior mean stays the refit mean.
        let shift = centre - values.iter().sum::<f64>() / draws as f64;
        for value in values.iter_mut() {
            *value += shift;
        }
    }
}

/// Where a fitted-model interval stands against the coverage measurements of its route.
pub(super) struct MechanismRefitScope {
    /// The measured statement the interval falls under, when it does.
    pub(super) statement: Option<&'static str>,
    /// Why the interval is outside every measured statement; empty inside one.
    pub(super) outside: String,
}

impl MechanismRefitScope {
    fn inside(statement: &'static str) -> Self {
        Self { statement: Some(statement), outside: String::new() }
    }

    fn outside(reason: impl Into<String>) -> Self {
        Self { statement: None, outside: reason.into() }
    }

    /// The scope as an informational diagnostic of the result.
    pub(super) fn diagnostic(&self) -> Diagnostic {
        let (message, fields) = match self.statement {
            Some(statement) => (
                format!(
                    "the published interval is inside the conditions its coverage measurement \
                     ran under ({statement})"
                ),
                vec![
                    ("coverage_scope".to_owned(), "measured_conditions".to_owned()),
                    ("statement".to_owned(), statement.to_owned()),
                ],
            ),
            None => (
                format!(
                    "no coverage measurement covers the published interval: {}. The route \
                     states no support for it",
                    self.outside
                ),
                vec![
                    ("coverage_scope".to_owned(), "not_measured".to_owned()),
                    ("reason".to_owned(), self.outside.clone()),
                ],
            ),
        };
        Diagnostic::new(
            "gcm.model.interval_scope",
            super::DiagnosticKind::Scientific,
            super::DiagnosticSeverity::Info,
            message,
        )
        .with_fields(fields)
    }
}

impl CheckedAttributionOperation {
    /// The independently weighted units behind a refit of `query` on `data`: rows of a static
    /// table, blocks of consecutive times on a lag-aligned unfolding. A change target weights
    /// its two populations separately, so it reports the smaller population's units and two
    /// populations; `None` when a population cannot be resolved.
    pub(super) fn mechanism_refit_units(
        &self,
        query: &CausalQuery,
        data: &TabularData,
    ) -> Option<(usize, usize)> {
        use antecedent_core::AttributeTarget;
        let units = |rows: usize| -> usize {
            if self.lag_aligned.is_some() {
                rows.div_ceil(self.temporal_block_length(rows).max(1))
            } else {
                rows
            }
        };
        let change = match query {
            CausalQuery::Attribute(attribute) => match &attribute.target {
                AttributeTarget::DistributionChange { change, .. }
                | AttributeTarget::RootCauses { change, .. }
                | AttributeTarget::StructureChange { change, .. } => Some(change),
                _ => None,
            },
            _ => None,
        };
        match change {
            Some(change) => {
                let baseline =
                    antecedent_attribution::resolve_rows(data, &change.baseline).ok()?.len();
                let comparison =
                    antecedent_attribution::resolve_rows(data, &change.comparison).ok()?.len();
                Some((units(baseline).min(units(comparison)), 2))
            }
            None => Some((units(data.row_count()), 1)),
        }
    }

    /// Where the uncertainty statement this execution publishes for `query` stands against the
    /// measured statements of its route (the target, the resampling, the draw or replicate
    /// count and the smallest measured support), or `None` when it publishes a point alone.
    #[allow(
        clippy::too_many_lines,
        reason = "one table of the measured statements of the fitted-model routes"
    )]
    pub(super) fn mechanism_refit_scope(
        &self,
        query: &CausalQuery,
        data: &TabularData,
    ) -> Option<MechanismRefitScope> {
        use antecedent_core::AttributeTarget;
        let temporal = self.lag_aligned.is_some();
        if self.procedure == AttributionProcedure::Point {
            let CausalQuery::Attribute(attribute) = query else {
                return None;
            };
            return match &attribute.target {
                // The whole-procedure interval of an intrinsic influence is in the payload.
                AttributeTarget::IntrinsicInfluence(target)
                    if target.fit_bootstrap_replicates > 0 =>
                {
                    Some(if temporal {
                        MechanismRefitScope::outside(
                            "the block-bootstrap whole-procedure interval has no coverage \
                             measurement",
                        )
                    } else if target.fit_bootstrap_replicates < 40 {
                        MechanismRefitScope::outside(format!(
                            "{} whole-procedure refits, where the measurement ran 40",
                            target.fit_bootstrap_replicates
                        ))
                    } else if data.row_count() < 150 {
                        MechanismRefitScope::outside(format!(
                            "{} rows, where the measurement starts at 150",
                            data.row_count()
                        ))
                    } else {
                        MechanismRefitScope::inside(
                            "whole-procedure Student-t interval of the intrinsic influence of \
                             each noise source, at least 40 refits and 150 rows",
                        )
                    })
                }
                // The posterior probability of a change under the declared prior.
                AttributeTarget::MechanismChange { query: change, .. }
                    if self.inference.is_bayesian() && change.change_prior.is_some() =>
                {
                    let least = antecedent_attribution::resolve_rows(data, &change.baseline)
                        .ok()
                        .zip(antecedent_attribution::resolve_rows(data, &change.comparison).ok())
                        .map(|(baseline, comparison)| baseline.len().min(comparison.len()));
                    Some(match least {
                        _ if temporal => MechanismRefitScope::outside(
                            "the posterior probability of a change on a series has no \
                             coverage measurement",
                        ),
                        Some(rows) if rows >= 75 => MechanismRefitScope::inside(
                            "credible set of the change indicator from the posterior \
                             probability of a change, held to at least nominal coverage, at \
                             least 75 rows per population",
                        ),
                        Some(rows) => MechanismRefitScope::outside(format!(
                            "{rows} rows in the smaller population, where the measurement \
                             starts at 75"
                        )),
                        None => MechanismRefitScope::outside(
                            "a population of the target could not be read",
                        ),
                    })
                }
                _ => None,
            };
        }
        Some(self.mechanism_refit_interval_scope(query, data))
    }

    /// [`Self::mechanism_refit_scope`] for a posterior or a refit bootstrap.
    #[allow(
        clippy::too_many_lines,
        reason = "one table of the measured statements of the fitted-model routes"
    )]
    fn mechanism_refit_interval_scope(
        &self,
        query: &CausalQuery,
        data: &TabularData,
    ) -> MechanismRefitScope {
        use antecedent_core::{
            AnomalyReference, AttributeTarget, ChangeEstimator, SimulationTarget,
        };
        let temporal = self.lag_aligned.is_some();
        let (bayesian, count_ok, count_reason) = match self.procedure {
            AttributionProcedure::SharedDirichletRowWeights { draws } => (
                true,
                draws >= MEASURED_POSTERIOR_DRAWS,
                format!(
                    "{draws} posterior draws, where the measurement ran \
                     {MEASURED_POSTERIOR_DRAWS}"
                ),
            ),
            AttributionProcedure::RowBootstrap { replicates } => (
                false,
                replicates >= MEASURED_BOOTSTRAP_REPLICATES,
                format!(
                    "{replicates} bootstrap replicates, where the measurement ran \
                     {MEASURED_BOOTSTRAP_REPLICATES}"
                ),
            ),
            AttributionProcedure::Point => {
                return MechanismRefitScope::outside("the execution publishes no interval");
            }
        };
        let Some((units, _)) = self.mechanism_refit_units(query, data) else {
            return MechanismRefitScope::outside("a population of the target could not be read");
        };
        let rows = data.row_count();
        // (statement, least units, further condition and its reason).
        let measured: Option<(&'static str, usize, bool, &'static str)> = match query {
            CausalQuery::Attribute(attribute) => match &attribute.target {
                AttributeTarget::DistributionChange {
                    estimator: ChangeEstimator::Sampled { n_samples, .. },
                    ..
                }
                | AttributeTarget::RootCauses {
                    estimator: ChangeEstimator::Sampled { n_samples, .. },
                    ..
                } => {
                    let enough = *n_samples >= 2_000;
                    let reason = "fewer than 2000 samples of the change estimator";
                    match (temporal, bayesian) {
                        (false, true) => Some((
                            "row-weight posterior of the total and mechanism components of a \
                             sampled mean-difference change, at least 300 rows per population",
                            300,
                            enough,
                            reason,
                        )),
                        (false, false) => Some((
                            "row-bootstrap normal interval of the total of a sampled \
                             mean-difference change, at least 300 rows per population",
                            300,
                            enough,
                            reason,
                        )),
                        (true, true) => Some((
                            "block-weight posterior of the total of a sampled mean-difference \
                             change on one series, at least 43 blocks per population",
                            43,
                            enough,
                            reason,
                        )),
                        (true, false) => Some((
                            "circular-block bootstrap normal interval of the total of a \
                             sampled mean-difference change on one series, at least 43 blocks \
                             per population",
                            43,
                            enough,
                            reason,
                        )),
                    }
                }
                AttributeTarget::Paths { .. } if bayesian && !temporal => Some((
                    "row-weight posterior of the path decomposition, at least 300 rows",
                    300,
                    true,
                    "",
                )),
                AttributeTarget::FeatureRelevance { n_samples, .. } if bayesian && !temporal => {
                    Some((
                        "row-weight posterior of the feature relevance scores at the observed \
                         feature means, at least 300 rows",
                        300,
                        *n_samples >= 50_000,
                        "fewer than 50000 samples per coalition value",
                    ))
                }
                AttributeTarget::Anomaly(anomaly) if bayesian && !temporal => Some((
                    "row-weight posterior of the mean anomaly score against a fixed \
                     reference, at least 300 rows",
                    300,
                    matches!(anomaly.reference, AnomalyReference::Fixed { .. }),
                    "the anomaly reference is estimated from the rows",
                )),
                AttributeTarget::IntrinsicInfluence(_) if bayesian && !temporal => Some((
                    "row-weight posterior of the intrinsic influence of each noise source, \
                     held to at least nominal coverage, at least 150 rows",
                    150,
                    true,
                    "",
                )),
                _ => None,
            },
            CausalQuery::Simulate(simulate) => match &simulate.target {
                SimulationTarget::PolicyValue { draws, .. } if !temporal => Some((
                    if bayesian {
                        "row-weight posterior of the policy mean, the baseline mean and their \
                         difference, at least 250 rows"
                    } else {
                        "row-bootstrap normal interval of the policy mean difference, at \
                         least 250 rows"
                    },
                    250,
                    *draws >= rows.saturating_mul(50),
                    "fewer than 50 simulated draws per fitted row",
                )),
                SimulationTarget::TemporalRollout { n_rows, .. } if temporal && bayesian => Some((
                    "block-weight posterior of the per-step means of a temporal rollout, at \
                     least 25 blocks",
                    25,
                    n_rows.saturating_mul(6) >= rows.saturating_mul(1_000),
                    "fewer than 1000 simulated rollout rows per 6 fitted rows",
                )),
                _ => None,
            },
            _ => None,
        };
        let Some((statement, least, condition, condition_reason)) = measured else {
            return MechanismRefitScope::outside(
                "this target under this resampling has no coverage measurement",
            );
        };
        if !count_ok {
            return MechanismRefitScope::outside(count_reason);
        }
        if units < least {
            return MechanismRefitScope::outside(format!(
                "{units} independently weighted {} in the smallest population, where the \
                 measurement starts at {least}",
                if temporal { "blocks" } else { "rows" }
            ));
        }
        if !condition {
            return MechanismRefitScope::outside(condition_reason);
        }
        MechanismRefitScope::inside(statement)
    }

    /// The posterior over the mechanisms of a fitted-model target: the Dirichlet-weighted
    /// refits of the attribution referred to their finite-sample reference, or the refit
    /// contributions the intrinsic-influence engine returned.
    pub(super) fn model_posterior(
        &self,
        query: &CausalQuery,
        data: &TabularData,
        answer: &super::super::structural_route::ModelAnswer,
        draws: usize,
        observed: Option<&super::super::temporal_attribution::CompleteWindowInput>,
        ctx: &ExecutionContext,
    ) -> Result<CausalPosterior, CausalError> {
        let (columns, has_primary_effect) = if attribute_intrinsic_target(query) {
            (intrinsic_posterior_columns(&answer.slot)?, false)
        } else {
            let replicates = self.attribute_replicates(query, data, draws, true, observed, ctx)?;
            let primary = replicates.names.first().is_some_and(|name| name == "total_change");
            let mut columns = replicates.columns();
            if let Some((units, populations)) = self.mechanism_refit_units(query, data) {
                let mut reference =
                    ctx.rng.stream_for(antecedent_core::StreamDomain::Attribution, 0xA77B_7EF5);
                mechanism_refit_reference(&mut columns, units, populations, &mut reference);
            }
            (columns, primary)
        };
        super::super::attribution_path::attribution_posterior(
            columns,
            self.cache.identification.required_assumptions.clone(),
            self.cache.identification.status,
            super::super::ATTRIBUTE_DIRICHLET_BACKEND,
            has_primary_effect,
        )
    }

    /// The mechanism-refit bootstrap of a fitted-model target: the standard error of the
    /// quantity the estimate's point is on the estimate, and every quantity's standard error in
    /// a diagnostic.
    pub(super) fn model_bootstrap(
        &self,
        query: &CausalQuery,
        data: &TabularData,
        replicates: u32,
        answer: &mut super::super::structural_route::ModelAnswer,
        observed: Option<&super::super::temporal_attribution::CompleteWindowInput>,
        ctx: &ExecutionContext,
    ) -> Result<(), CausalError> {
        let draws = usize::try_from(replicates).unwrap_or(usize::MAX);
        let outcome = self.attribute_replicates(query, data, draws, false, observed, ctx)?;
        let diagnostic = disclose_model_bootstrap(
            &outcome,
            replicates,
            published_quantity(query),
            &mut answer.estimate,
            |target, estimate| self.bootstrap_disclosure(target, data.row_count(), estimate),
        );
        answer.diagnostics.push(diagnostic);
        Ok(())
    }

    /// Recompute a fitted-model attribution on `replicates` refits of the mechanisms: each
    /// under fresh Dirichlet (row or block) weights when `bayesian`, else fresh bootstrap
    /// counts. The mechanism families are the ones selected on the original rows, as on the
    /// anomaly and change cells.
    ///
    /// A failed replicate is an error under weights and failed mass under counts; it is never
    /// redrawn.
    pub(super) fn attribute_replicates(
        &self,
        query: &CausalQuery,
        data: &TabularData,
        replicates: usize,
        bayesian: bool,
        observed: Option<&super::super::temporal_attribution::CompleteWindowInput>,
        ctx: &ExecutionContext,
    ) -> Result<Replicates, CausalError> {
        let (attribute, simulate) = match query {
            CausalQuery::Attribute(attribute) => (Some(attribute), None),
            CausalQuery::Simulate(simulate) => (None, Some(simulate)),
            _ => {
                return Err(CausalError::Compile {
                    message: "a mechanism-refit replicate needs an attribution or rollout query"
                        .into(),
                });
            }
        };
        let fitted = self.mechanisms.fit(&self.graph, data)?;
        let mut stream =
            ctx.rng.stream_for(antecedent_core::StreamDomain::Attribution, 0xA77B_7A7E);
        let mut out = Replicates { names: Vec::new(), rows: Vec::new(), failed: 0 };
        let mut admitted = None;
        for replicate in 0..replicates {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled {
                    stage: super::super::super::stage::STAGE_UNCERTAINTY,
                });
            }
            let mut counts_stream;
            let rng = if bayesian {
                &mut stream
            } else {
                counts_stream = super::super::gcm_bootstrap::replicate_stream(
                    ctx,
                    0xA77B_B007,
                    u32::try_from(replicate).unwrap_or(u32::MAX),
                );
                &mut counts_stream
            };
            let replicate = match (attribute, simulate) {
                (Some(attribute), _) => self.attribute_replicate(
                    attribute,
                    &fitted,
                    data,
                    bayesian,
                    rng,
                    replicate as u64,
                    ctx,
                ),
                (None, Some(simulate)) => self.simulate_replicate(
                    simulate,
                    &fitted,
                    data,
                    bayesian,
                    rng,
                    replicate as u64,
                    observed,
                    ctx,
                ),
                (None, None) => unreachable!("a replicate query is an attribution or a rollout"),
            };
            match replicate {
                Ok(values) => {
                    if out.rows.is_empty() {
                        out.names = values.iter().map(|(name, _)| name.clone()).collect();
                        admitted = Some(admit_posterior_columns(ctx, replicates, out.names.len())?);
                    } else if values.len() != out.names.len()
                        || values.iter().zip(&out.names).any(|((name, _), first)| name != first)
                    {
                        return Err(CausalError::Compile {
                            message: "a mechanism-refit replicate changed the attribution's \
                                      components"
                                .into(),
                        });
                    }
                    out.rows.push(values.into_iter().map(|(_, value)| value).collect());
                }
                Err(_) if ctx.cancellation.is_cancelled() => {
                    return Err(CausalError::Cancelled {
                        stage: super::super::super::stage::STAGE_UNCERTAINTY,
                    });
                }
                Err(error @ (CausalError::Cancelled { .. } | CausalError::Resource { .. }))
                    if self.temporal_input.is_some() =>
                {
                    return Err(error);
                }
                Err(error) if bayesian => return Err(error),
                Err(_) => out.failed += 1,
            }
        }
        drop(admitted);
        if out.rows.len() < 2 && bayesian {
            return Err(CausalError::Compile {
                message: "Bayesian attribution draws have inconsistent shape".into(),
            });
        }
        Ok(out)
    }

    /// One replicate of a temporal rollout: the mechanisms refit under fresh block weights (or
    /// block counts), the dynamics rolled forward from the last rows of the series, and the mean
    /// of every variable at every step read off the rollout.
    #[allow(
        clippy::too_many_lines,
        clippy::too_many_arguments,
        reason = "the bounded operation keeps its execution inputs and resource accounting together"
    )]
    pub(super) fn simulate_replicate(
        &self,
        simulate: &antecedent_core::SimulateQuery,
        fitted: &crate::gcm::FittedGcm,
        data: &TabularData,
        bayesian: bool,
        rng: &mut antecedent_core::CausalRng,
        replicate: u64,
        observed: Option<&super::super::temporal_attribution::CompleteWindowInput>,
        ctx: &ExecutionContext,
    ) -> Result<Vec<(String, f64)>, CausalError> {
        let n = data.row_count();
        let weights = if bayesian {
            self.draw_row_weights(n, rng, self.full_temporal_runs(), replicate, ctx)?
        } else {
            self.draw_row_counts(n, rng, self.full_temporal_runs(), replicate, ctx)?
        };
        let registry = crate::gcm::MechanismRegistry::standard();
        let cancelled = || ctx.cancellation.is_cancelled();
        let store = if bayesian {
            registry.refit_weighted_polled(
                &fitted.model,
                data,
                &fitted.assignments,
                &weights,
                &cancelled,
            )
        } else {
            registry.refit_frequency_weighted_polled(
                &fitted.model,
                data,
                &fitted.assignments,
                &weights,
                &cancelled,
            )
        }
        .map_err(|error| {
            if self.temporal_input.is_some() {
                CausalError::from(error)
            } else {
                CausalError::rejected(error)
            }
        })?;
        let refit = crate::gcm::FittedGcm {
            model: fitted.model.clone().with_mechanisms(store),
            assignments: fitted.assignments.clone(),
        };
        let outcome = match &self.lag_aligned {
            Some(aligned) => super::super::structural_route::rollout_on_unfolding(
                &refit,
                data,
                aligned,
                observed,
                self.temporal_budget.as_ref(),
                simulate,
                ctx,
            )?,
            None => crate::analysis::structure_route::run_simulate(&refit, data, simulate, ctx)?,
        };
        let nodes = self.lag_aligned.as_ref().map_or(
            refit.model.n_nodes(),
            super::super::temporal_attribution::LagAlignedScm::contemporaneous_count,
        );
        let mut quantities = crate::analysis::structure_route::simulation_quantities(
            &outcome, nodes,
        )
        .ok_or_else(|| CausalError::Compile {
            message: "this simulation has no mechanism-refit replicate".into(),
        })?;
        // The means above reuse the query's seed in every refit (common random numbers keep the
        // refit spread free of Monte Carlo noise). The predictive law of a path needs fresh
        // innovations in every refit, so one path is rolled out under a seed drawn per replicate.
        if let antecedent_core::SimulationTarget::TemporalRollout {
            schedule,
            n_rows,
            n_steps,
            window,
            history,
            ..
        } = &simulate.target
        {
            let single = antecedent_core::SimulateQuery::new(
                antecedent_core::SimulationTarget::TemporalRollout {
                    schedule: schedule.clone(),
                    // A supplied history has one value per row, so it keeps its row count.
                    n_rows: if history.is_empty() { 1 } else { *n_rows },
                    n_steps: *n_steps,
                    window: *window,
                    history: history.clone(),
                    seed: rng.next_u64(),
                },
            );
            let path_outcome = match &self.lag_aligned {
                Some(aligned) => super::super::structural_route::rollout_on_unfolding(
                    &refit,
                    data,
                    aligned,
                    observed,
                    self.temporal_budget.as_ref(),
                    &single,
                    ctx,
                )?,
                None => crate::analysis::structure_route::run_simulate(&refit, data, &single, ctx)?,
            };
            quantities.extend(
                crate::analysis::structure_route::simulated_path_quantities(&path_outcome, nodes)
                    .ok_or_else(|| CausalError::Compile {
                    message: "a rollout replicate has no simulated path".into(),
                })?,
            );
        }
        Ok(quantities)
    }

    /// One replicate of a fitted-model attribution on mechanisms refit under fresh weights.
    #[allow(
        clippy::too_many_lines,
        clippy::too_many_arguments,
        reason = "the bounded operation keeps its execution inputs and resource accounting together"
    )]
    pub(super) fn attribute_replicate(
        &self,
        attribute: &antecedent_core::AttributeQuery,
        fitted: &crate::gcm::FittedGcm,
        data: &TabularData,
        bayesian: bool,
        rng: &mut antecedent_core::CausalRng,
        replicate: u64,
        ctx: &ExecutionContext,
    ) -> Result<Vec<(String, f64)>, CausalError> {
        use antecedent_core::{AttributeTarget, ChangeEstimator};
        let draw = |op: &Self,
                    n: usize,
                    positions: Option<&[usize]>,
                    rng: &mut antecedent_core::CausalRng,
                    key: u64| {
            let partition =
                positions.map(|rows| op.selected_temporal_runs(rows, ctx)).transpose()?.flatten();
            let runs = partition
                .as_ref()
                .map(super::super::temporal_attribution::SelectedRuns::ranges)
                .or_else(|| if positions.is_none() { op.full_temporal_runs() } else { None });
            if bayesian {
                op.draw_row_weights(n, rng, runs, key, ctx)
            } else {
                op.draw_row_counts(n, rng, runs, key, ctx)
            }
        };
        let draw_populations = |change: &antecedent_core::ChangeAttributionQuery,
                                rng: &mut antecedent_core::CausalRng|
         -> Result<(SampledWeights, SampledWeights), CausalError> {
            let baseline_rows = antecedent_attribution::resolve_rows(data, &change.baseline)
                .map_err(CausalError::rejected)?;
            let comparison_rows = antecedent_attribution::resolve_rows(data, &change.comparison)
                .map_err(CausalError::rejected)?;
            let baseline = draw(
                self,
                baseline_rows.len(),
                Some(&baseline_rows),
                rng,
                antecedent_core::derive_stream(replicate, 0),
            )?;
            let comparison = draw(
                self,
                comparison_rows.len(),
                Some(&comparison_rows),
                rng,
                antecedent_core::derive_stream(replicate, 1),
            )?;
            Ok((baseline, comparison))
        };
        match &attribute.target {
            AttributeTarget::DistributionChange { change, estimator }
            | AttributeTarget::RootCauses { change, estimator } => {
                let (baseline, comparison) = draw_populations(change, rng)?;
                let refit = population_refit(bayesian, &baseline, &comparison);
                let result = match estimator {
                    ChangeEstimator::Sampled { measure, n_samples, seed } => {
                        let options = antecedent_attribution::DistributionChangeOptions {
                            measure: difference_measure(*measure),
                            n_samples: *n_samples,
                            seed: *seed,
                        };
                        if bayesian {
                            antecedent_attribution::distribution_change_with_row_weights(
                                &fitted.model,
                                data,
                                change,
                                &options,
                                &baseline,
                                &comparison,
                                ctx,
                            )
                        } else {
                            antecedent_attribution::distribution_change_with_frequency_weights(
                                &fitted.model,
                                data,
                                change,
                                &options,
                                &baseline,
                                &comparison,
                                ctx,
                            )
                        }
                    }
                    ChangeEstimator::Robust { max_rows, n_samples, seed } => {
                        antecedent_attribution::distribution_change_robust_with_refit(
                            &fitted.model,
                            data,
                            change,
                            &antecedent_attribution::RobustChangeOptions {
                                max_rows: *max_rows,
                                n_samples: *n_samples,
                                seed: *seed,
                            },
                            &refit,
                            ctx,
                        )
                    }
                }
                .map_err(CausalError::rejected)?;
                Ok(change_quantities(&result))
            }
            AttributeTarget::StructureChange { change, comparison, measure, n_samples, seed } => {
                let (baseline, comparison_weights) = draw_populations(change, rng)?;
                let refit = population_refit(bayesian, &baseline, &comparison_weights);
                let comparison_model =
                    crate::analysis::structure_route::structure_comparison_model(
                        &self.graph,
                        comparison,
                    )?;
                let result = antecedent_attribution::structure_change_with_refit(
                    &fitted.model,
                    &comparison_model,
                    data,
                    change,
                    &antecedent_attribution::StructureChangeOptions {
                        measure: difference_measure(*measure),
                        n_samples: *n_samples,
                        seed: *seed,
                    },
                    &refit,
                    ctx,
                )
                .map_err(CausalError::rejected)?;
                Ok(change_quantities(&result))
            }
            AttributeTarget::MechanismChange { .. } => Err(CausalError::Unsupported {
                message: "mechanism-change detection is a test, not a smooth functional of the \
                          refitted mechanisms",
            }),
            AttributeTarget::IntrinsicInfluence(_) => Err(CausalError::Compile {
                message: "intrinsic influence draws its refits inside the engine".into(),
            }),
            _ => {
                // Paths, feature relevance, anomaly scores and unit change read the fitted
                // mechanisms (and, for the last three, the rows): refit them on the whole
                // table under one weight vector and recompute.
                let weights = draw(self, data.row_count(), None, rng, replicate)?;
                self.whole_table_replicate(attribute, fitted, data, bayesian, &weights, ctx)
            }
        }
    }

    /// The quantities of an attribution recomputed on mechanisms refit on the whole table under
    /// one weight vector.
    pub(super) fn whole_table_replicate(
        &self,
        attribute: &antecedent_core::AttributeQuery,
        fitted: &crate::gcm::FittedGcm,
        data: &TabularData,
        bayesian: bool,
        weights: &[f64],
        ctx: &ExecutionContext,
    ) -> Result<Vec<(String, f64)>, CausalError> {
        let registry = crate::gcm::MechanismRegistry::standard();
        let store = if bayesian {
            registry.refit_weighted_polled(
                &fitted.model,
                data,
                &fitted.assignments,
                weights,
                &|| ctx.cancellation.is_cancelled(),
            )
        } else {
            registry.refit_frequency_weighted_polled(
                &fitted.model,
                data,
                &fitted.assignments,
                weights,
                &|| ctx.cancellation.is_cancelled(),
            )
        }
        .map_err(|error| {
            if self.temporal_input.is_some() {
                CausalError::from(error)
            } else {
                CausalError::rejected(error)
            }
        })?;
        let refit = crate::gcm::FittedGcm {
            model: fitted.model.clone().with_mechanisms(store),
            assignments: fitted.assignments.clone(),
        };
        let outcome = crate::analysis::structure_route::run_attribute_with_fit(
            &refit,
            &self.graph,
            data,
            attribute,
            None,
            crate::analysis::structure_route::ChangeReading::default(),
            None,
            None,
            ctx,
        )?;
        attribution_quantities_weighted(&outcome, weights).ok_or_else(|| CausalError::Compile {
            message: "this attribution has no mechanism-refit replicate".into(),
        })
    }

    /// Draw count published on the result: absent for point procedures.
    pub(super) fn published_draws(&self) -> Option<u32> {
        match self.procedure {
            AttributionProcedure::SharedDirichletRowWeights { draws } => {
                Some(u32::try_from(draws).unwrap_or(u32::MAX))
            }
            AttributionProcedure::Point | AttributionProcedure::RowBootstrap { .. } => None,
        }
    }

    /// One frequentist replicate's row counts over `n` rows: multinomial single rows on a
    /// static table, circular blocks of consecutive times on a lag-aligned unfolding.
    pub(super) fn draw_row_counts(
        &self,
        n: usize,
        rng: &mut antecedent_core::CausalRng,
        runs: Option<&[std::ops::Range<usize>]>,
        replicate: u64,
        ctx: &ExecutionContext,
    ) -> Result<SampledWeights, CausalError> {
        match (&self.lag_aligned, runs, &self.temporal_budget) {
            (Some(_), Some(runs), Some(budget)) => {
                let mut budget = budget.clone();
                antecedent_attribution::temporal_resampling::bootstrap_run_counts(
                    n,
                    self.temporal_block_length(n),
                    runs,
                    replicate,
                    rng,
                    &mut budget,
                    ctx,
                )
                .map(SampledWeights::Run)
                .map_err(CausalError::from)
            }
            (Some(_), None, _) => Err(CausalError::Compile {
                message: "temporal resampling lost its row partition".into(),
            }),
            (None, _, _) => {
                Ok(SampledWeights::Legacy(super::super::gcm_bootstrap::bootstrap_counts(rng, n)))
            }
            _ => Err(CausalError::Compile {
                message: "temporal resampling lost its shared budget".into(),
            }),
        }
    }

    pub(super) fn selected_temporal_runs(
        &self,
        positions: &[usize],
        ctx: &ExecutionContext,
    ) -> Result<Option<super::super::temporal_attribution::SelectedRuns>, CausalError> {
        match (&self.temporal_input, &self.temporal_budget) {
            (Some(input), Some(budget)) => {
                let mut budget = budget.clone();
                Ok(Some(input.selected_runs(positions, &mut budget, ctx)?))
            }
            (None, None) => Ok(None),
            _ => Err(CausalError::Compile {
                message: "temporal resampling lost its row owner".into(),
            }),
        }
    }

    pub(super) fn temporal_block_length(&self, n: usize) -> usize {
        self.temporal_input.as_ref().map_or_else(
            || self.lag_aligned.as_ref().map_or(1, |aligned| aligned.block_length(n)),
            |input| input.block_length(n),
        )
    }

    pub(super) fn full_temporal_runs(&self) -> Option<&[std::ops::Range<usize>]> {
        self.temporal_input
            .as_ref()
            .map(super::super::temporal_attribution::CompleteWindowInput::runs)
    }

    /// The bootstrap disclosure of `target` over `n` rows, recording the row-block
    /// family on `estimate` when the rows are one lag-aligned series.
    pub(super) fn bootstrap_disclosure(
        &self,
        target: &str,
        n: usize,
        estimate: &mut EffectEstimate,
    ) -> Diagnostic {
        let accounting = estimate.uncertainty.bootstrap;
        match &self.lag_aligned {
            Some(_) => {
                estimate.uncertainty.block.family =
                    Some(antecedent_estimate::CircularBlockFamily::RowBlock);
                super::super::gcm_bootstrap::bootstrap_diagnostic_over(
                    target,
                    &format!(
                        "v2 run-aware circular-block bootstrap (nominal block {}; disconnected segments use derived streams and wrap only within their own run)",
                        self.temporal_block_length(n)
                    ),
                    &accounting,
                )
            }
            None => super::super::gcm_bootstrap::bootstrap_diagnostic(target, &accounting),
        }
    }

    /// One draw's row weights over `n` rows: Dirichlet row weights on a static table,
    /// block weights on the consecutive times of a lag-aligned unfolding.
    pub(super) fn draw_row_weights(
        &self,
        n: usize,
        rng: &mut antecedent_core::CausalRng,
        runs: Option<&[std::ops::Range<usize>]>,
        replicate: u64,
        ctx: &ExecutionContext,
    ) -> Result<SampledWeights, CausalError> {
        match (&self.lag_aligned, runs, &self.temporal_budget) {
            (Some(_), Some(runs), Some(budget)) => {
                let mut budget = budget.clone();
                antecedent_attribution::temporal_resampling::bayesian_run_weights(
                    n,
                    self.temporal_block_length(n),
                    runs,
                    replicate,
                    rng,
                    &mut budget,
                    ctx,
                )
                .map(SampledWeights::Run)
                .map_err(CausalError::from)
            }
            (Some(_), None, _) => Err(CausalError::Compile {
                message: "temporal resampling lost its row partition".into(),
            }),
            (None, _, _) => Ok(SampledWeights::Legacy(
                super::super::attribution_path::dirichlet_row_weights(n, rng),
            )),
            _ => Err(CausalError::Compile {
                message: "temporal resampling lost its shared budget".into(),
            }),
        }
    }
}
