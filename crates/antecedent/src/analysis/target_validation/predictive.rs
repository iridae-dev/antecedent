//! Prior-sensitivity and posterior-predictive target checks.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalRng, DenseLinearAlgebra, ExecutionContext, FaerBackend, InferenceMode,
    LeastSquaresWorkspace, Modal, PanelData, PanelUnit, PosteriorPredictiveTarget, PreparedRefit,
    PreparedTargetData, PriorSensitivityTarget, TargetRefitError, TargetRefutationReport,
    TargetRefuterKind, TimeSeriesData, ValidationError, admit_target_cells, data_failed,
    gram_cholesky_factor, max_graph_lag, posterior_predictive_target_check,
    prior_sensitivity_target_check, refuter_outcome, sample_gamma, scale_prior_deviations,
    series_with, target_refuter_stream, unsupported,
};

impl PreparedRefit<'_> {
    /// Prior sensitivity through the compiler: the study prepared under each alternative
    /// isotropic prior scale, the posterior of the whole target read back from each.
    pub(super) fn prior_sensitivity(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<TargetRefutationReport, TargetRefitError> {
        let InferenceMode::Bayesian(config) = &self.study().settings.inference else {
            return Err(unsupported(
                "prior sensitivity reads a posterior; the study is frequentist",
                "prepare the study with Bayesian inference",
            ));
        };
        // The static response derivative posterior is a Rubin row-weight
        // bootstrap. Its disclosure is attached by the estimator itself; it
        // has no coefficient prior whose scale can be perturbed. A successful
        // rerun at another unused scale is not prior-sensitivity evidence.
        if self.result.response().is_some_and(|response| {
            response.assumptions.entries.iter().any(|record| {
                matches!(&record.source, antecedent_core::AssumptionSource::AlgorithmDefault { algorithm }
                    if algorithm.as_ref() == "response.bayesian.derivative")
            })
        }) {
            return Err(unsupported(
                "the Rubin Bayesian-bootstrap derivative does not use a coefficient prior",
                "use a posterior route with an actually varied coefficient prior",
            ));
        }
        let baseline = self.target_draws(self.result).ok_or_else(|| {
            unsupported(
                "the result retains no posterior draws of its whole target",
                "a Bayesian route that retains target draws",
            )
        })?;
        let mut alternatives: Vec<(Arc<str>, Vec<f64>)> = Vec::new();
        for &factor in &self.options.prior_scale_factors {
            if !(factor.is_finite() && factor > 0.0) {
                continue;
            }
            let mut study = self.study().clone();
            let mut alternative = config.clone();
            // An explicit coefficient prior has no single scale: its Gaussian coefficient
            // standard deviations are multiplied by `factor` around their own means (variances
            // by `factor^2`), the same perturbation the isotropic scale undergoes.
            // A prior hydrated from a posterior artifact, or composed from external sources,
            // exists only once the design is bound, so its variances are scaled where the route
            // resolves it (`prior_variance_multiplier`), not here.
            match &config.prior {
                Some(prior) => alternative.prior = Some(scale_prior_deviations(prior, factor)?),
                None if config.prior_artifact.is_some() || config.external_compose.is_some() => {
                    alternative.prior_variance_multiplier = Some(factor * factor);
                }
                None => alternative.prior_scale *= factor,
            }
            study.settings.inference = InferenceMode::Bayesian(alternative);
            let refitted =
                Self::compile_and_run(study, &self.modal_data(), ctx).map_err(|error| {
                    unsupported(
                        format!("the study under prior scale x{factor} did not run: {error}"),
                        "a Bayesian route whose prior scale can be varied",
                    )
                })?;
            let draws = self.target_draws(&refitted).ok_or_else(|| {
                unsupported(
                    "the refit under the alternative prior retains no target draws",
                    "a Bayesian route that retains target draws",
                )
            })?;
            alternatives.push((Arc::from(format!("prior_scale x{factor}")), draws));
        }
        if alternatives.is_empty() {
            return Err(unsupported("no alternative prior scale was declared", "declare one"));
        }
        prior_sensitivity_target_check(&PriorSensitivityTarget {
            target: &self.target,
            baseline: &baseline,
            alternatives: &alternatives,
            tolerance: self.options.prior_shift_tolerance,
        })
        .map_err(refuter_outcome)
    }

    /// Posterior-predictive check of the target through the compiler, under the reference
    /// linear-Gaussian outcome model of the treatment and adjustment set: for each draw
    /// `(β_s, σ_s)` of its reference posterior the program is run on the noiseless outcome
    /// `μ_s = Xβ_s` (the target of the draw) and on a replicated outcome `μ_s + σ_s ε`; the
    /// observed target is compared with the replicated ones.
    #[allow(
        clippy::too_many_lines,
        reason = "the fit, its Cholesky factor and the draws share one reference design in sequence"
    )]
    pub(super) fn posterior_predictive(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<TargetRefutationReport, TargetRefitError> {
        let PreparedTargetData::Tabular(data) = self.data else {
            return self.posterior_predictive_lagged(ctx);
        };
        let storage = data.storage();
        if storage.analysis_mask().is_some() || storage.weights().is_some() {
            return Err(unsupported(
                "the reference outcome model replicates an unmasked, unweighted table",
                "refute the unmasked table",
            ));
        }
        let mut ids = vec![self.treatment];
        ids.extend(self.reference_covariates());
        let mut columns = Vec::with_capacity(ids.len());
        for id in &ids {
            columns.push(Self::column_of(storage, *id)?);
        }
        let y = Self::column_of(storage, self.outcome)?;
        let n = y.len();
        let p = ids.len() + 1;
        if n <= p + 2 || y.iter().chain(columns.iter().copied().flatten()).any(|v| !v.is_finite()) {
            return Err(unsupported(
                "the reference outcome model needs complete float columns and more rows than \
                 parameters",
                "resolve missing values first",
            ));
        }
        let mut design = vec![1.0; n];
        for column in &columns {
            design.extend_from_slice(column);
        }
        let fit = FaerBackend
            .least_squares(&design, n, p, y, &mut LeastSquaresWorkspace::default())
            .map_err(|error| {
                unsupported(
                    format!("the reference outcome regression could not be fit: {error}"),
                    "a full-rank reference design",
                )
            })?;
        let l = gram_cholesky_factor(
            &design,
            n,
            p,
            "the treatment and adjustment design is rank deficient",
        )?;
        let k = self.target.dimension();
        let draws_wanted = (self.replicates as usize).max(k + 2);
        let _lease = admit_target_cells(ctx, draws_wanted.saturating_mul(2), n)?;
        let mut thetas = Vec::with_capacity(draws_wanted * k);
        let mut replicated = Vec::with_capacity(draws_wanted * k);
        let dof = (n - p) as f64;
        for s in 0..draws_wanted {
            if ctx.cancellation.is_cancelled() {
                return Err(TargetRefitError::Failed(ValidationError::Cancelled));
            }
            let mut rng = target_refuter_stream(
                TargetRefuterKind::PosteriorPredictive,
                u32::try_from(s).unwrap_or(u32::MAX),
                ctx,
            );
            let chi2 = sample_gamma(dof / 2.0, 0.5, &mut rng);
            let sigma = (fit.rss / chi2.max(f64::MIN_POSITIVE)).sqrt();
            // β = β̂ + σ L⁻ᵀ z.
            let z: Vec<f64> =
                (0..p).map(|_| antecedent_kernels::standard_normal(&mut rng)).collect();
            let mut w = z;
            for i in (0..p).rev() {
                let mut acc = w[i];
                for q in (i + 1)..p {
                    acc -= l[q * p + i] * w[q];
                }
                w[i] = acc / l[i * p + i];
            }
            let beta: Vec<f64> =
                fit.coefficients.iter().zip(&w).map(|(b, w)| b + sigma * w).collect();
            let mean: Vec<f64> =
                (0..n).map(|r| (0..p).map(|c| design[c * n + r] * beta[c]).sum()).collect();
            let noisy: Arc<[f64]> = mean
                .iter()
                .map(|m| m + sigma * antecedent_kernels::standard_normal(&mut rng))
                .collect();
            let run = |outcome: Arc<[f64]>| -> Result<Vec<f64>, TargetRefitError> {
                let table = data.with_replaced_float(self.outcome, outcome).map_err(data_failed)?;
                self.reestimate(&Modal::Tabular(table), ctx).map_err(|error| match error {
                    TargetRefitError::Failed(error) => unsupported(
                        format!("the program did not run on a reference-model replicate: {error}"),
                        "a route that admits replicated outcomes",
                    ),
                    other @ TargetRefitError::Unsupported { .. } => other,
                })
            };
            thetas.extend(run(mean.into())?);
            replicated.extend(run(noisy)?);
        }
        posterior_predictive_target_check(&PosteriorPredictiveTarget {
            target: &self.target,
            observed: &self.original,
            draws: &thetas,
            replicated: &replicated,
            alpha: self.alpha,
        })
        .map_err(refuter_outcome)
    }

    /// Posterior-predictive check on a series or the units of a panel, under a reference outcome
    /// model that carries the lag structure the route's graph declares: the outcome regressed on
    /// the treatment, the adjustment set and its own `L` previous values (`L` the longest lag of
    /// the graph, at least one), pooled over units. Each draw of its reference posterior replaces
    /// every unit's outcome by the model's recursion with no noise (the target of the draw) and
    /// with simulated noise (a replicate), the first `L` periods of a unit staying as observed,
    /// and the compiled program is re-run on both.
    #[allow(
        clippy::too_many_lines,
        clippy::needless_range_loop,
        clippy::cast_precision_loss,
        reason = "the pooled lagged design, its Cholesky factor and the recursive draws share one reference model in sequence"
    )]
    fn posterior_predictive_lagged(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<TargetRefutationReport, TargetRefitError> {
        let units: Vec<&TimeSeriesData> = match self.data {
            PreparedTargetData::Series(series) => vec![series],
            PreparedTargetData::Panel(panel) => panel.units().iter().map(|u| &u.series).collect(),
            PreparedTargetData::MultiEnv(environments) => {
                environments.environments().iter().collect()
            }
            PreparedTargetData::Tabular(_) => {
                return Err(unsupported("a table has no lag structure", "a series or panel"));
            }
        };
        let lags = max_graph_lag(self.study()).max(1);
        let mut ids = vec![self.treatment];
        ids.extend(self.reference_covariates());
        // The reference model is an autoregressive distributed-lag regression: the outcome on its
        // own lags and on the treatment and the adjustment covariates at every lag the graph
        // spans, so a treatment read at an earlier period (a pulse at a lag) is in the model the
        // target is recomputed under, not only its contemporaneous value.
        let width = lags + 1;
        let y_base = ids.len() * width;
        let p = y_base + 1 + lags;
        let mut outcomes: Vec<&[f64]> = Vec::with_capacity(units.len());
        let mut covariates: Vec<Vec<&[f64]>> = Vec::with_capacity(units.len());
        let mut rows = 0_usize;
        for unit in &units {
            let storage = unit.storage();
            if storage.analysis_mask().is_some() || storage.weights().is_some() {
                return Err(unsupported(
                    "the reference outcome model replicates unmasked, unweighted series",
                    "refute the unmasked series",
                ));
            }
            let y = Self::column_of(storage, self.outcome)?;
            let mut columns = Vec::with_capacity(ids.len());
            for id in &ids {
                columns.push(Self::column_of(storage, *id)?);
            }
            if y.len() <= lags
                || y.iter().chain(columns.iter().copied().flatten()).any(|v| !v.is_finite())
            {
                return Err(unsupported(
                    "the reference outcome model needs complete float columns longer than the lag",
                    "resolve missing values first",
                ));
            }
            rows += y.len() - lags;
            outcomes.push(y);
            covariates.push(columns);
        }
        if rows <= p + 2 {
            return Err(unsupported(
                "the reference outcome model needs more lag-aligned rows than parameters",
                "a longer series",
            ));
        }
        // Column-major design over the lag-aligned rows of every unit.
        let mut design = vec![0.0; rows * p];
        let mut response = Vec::with_capacity(rows);
        let mut row = 0;
        for (u, y) in outcomes.iter().enumerate() {
            for t in lags..y.len() {
                design[row] = 1.0;
                for (c, column) in covariates[u].iter().enumerate() {
                    for k in 0..width {
                        design[(1 + c * width + k) * rows + row] = column[t - k];
                    }
                }
                for j in 1..=lags {
                    design[(y_base + j) * rows + row] = y[t - j];
                }
                response.push(y[t]);
                row += 1;
            }
        }
        let fit = FaerBackend
            .least_squares(&design, rows, p, &response, &mut LeastSquaresWorkspace::default())
            .map_err(|error| {
                unsupported(
                    format!("the reference outcome regression could not be fit: {error}"),
                    "a full-rank reference design",
                )
            })?;
        let l = gram_cholesky_factor(
            &design,
            rows,
            p,
            "the lagged reference design is rank deficient",
        )?;
        let k = self.target.dimension();
        let draws_wanted = (self.replicates as usize).max(k + 2);
        let _lease = admit_target_cells(ctx, draws_wanted.saturating_mul(2), rows)?;
        let mut thetas = Vec::with_capacity(draws_wanted * k);
        let mut replicated = Vec::with_capacity(draws_wanted * k);
        let dof = (rows - p) as f64;
        for s in 0..draws_wanted {
            if ctx.cancellation.is_cancelled() {
                return Err(TargetRefitError::Failed(ValidationError::Cancelled));
            }
            let mut rng = target_refuter_stream(
                TargetRefuterKind::PosteriorPredictive,
                u32::try_from(s).unwrap_or(u32::MAX),
                ctx,
            );
            let chi2 = sample_gamma(dof / 2.0, 0.5, &mut rng);
            let sigma = (fit.rss / chi2.max(f64::MIN_POSITIVE)).sqrt();
            let mut w: Vec<f64> =
                (0..p).map(|_| antecedent_kernels::standard_normal(&mut rng)).collect();
            for i in (0..p).rev() {
                let mut acc = w[i];
                for q in (i + 1)..p {
                    acc -= l[q * p + i] * w[q];
                }
                w[i] = acc / l[i * p + i];
            }
            let beta: Vec<f64> =
                fit.coefficients.iter().zip(&w).map(|(b, w)| b + sigma * w).collect();
            let simulate = |noisy: bool, rng: &mut CausalRng| -> Vec<Arc<[f64]>> {
                outcomes
                    .iter()
                    .enumerate()
                    .map(|(u, y)| {
                        let mut path = y.to_vec();
                        for t in lags..y.len() {
                            let mut mean = beta[0];
                            for (c, column) in covariates[u].iter().enumerate() {
                                for k in 0..width {
                                    mean += beta[1 + c * width + k] * column[t - k];
                                }
                            }
                            for j in 1..=lags {
                                mean += beta[y_base + j] * path[t - j];
                            }
                            path[t] = if noisy {
                                mean + sigma * antecedent_kernels::standard_normal(rng)
                            } else {
                                mean
                            };
                        }
                        Arc::from(path)
                    })
                    .collect()
            };
            let mean_paths = simulate(false, &mut rng);
            let noisy_paths = simulate(true, &mut rng);
            let run = |paths: Vec<Arc<[f64]>>| -> Result<Vec<f64>, TargetRefitError> {
                let mut replaced = Vec::with_capacity(units.len());
                for (unit, path) in units.iter().zip(paths) {
                    replaced.push(series_with(unit, &[(self.outcome, path)], None, None)?);
                }
                let modal = match self.data {
                    PreparedTargetData::Panel(panel) => {
                        let mut members = Vec::with_capacity(replaced.len());
                        for (unit, series) in panel.units().iter().zip(replaced) {
                            members.push(PanelUnit { unit_id: unit.unit_id, series });
                        }
                        Modal::Panel(PanelData::try_new(members).map_err(data_failed)?)
                    }
                    PreparedTargetData::MultiEnv(_) => Modal::MultiEnv(
                        antecedent_data::MultiEnvironmentData::try_new(replaced)
                            .map_err(data_failed)?,
                    ),
                    _ => Modal::Series(replaced.remove(0)),
                };
                self.reestimate(&modal, ctx).map_err(|error| match error {
                    TargetRefitError::Failed(error) => unsupported(
                        format!("the program did not run on a reference-model replicate: {error}"),
                        "a route that admits replicated outcomes",
                    ),
                    other @ TargetRefitError::Unsupported { .. } => other,
                })
            };
            thetas.extend(run(mean_paths)?);
            replicated.extend(run(noisy_paths)?);
        }
        posterior_predictive_target_check(&PosteriorPredictiveTarget {
            target: &self.target,
            observed: &self.original,
            draws: &thetas,
            replicated: &replicated,
            alpha: self.alpha,
        })
        .map_err(refuter_outcome)
    }
}
