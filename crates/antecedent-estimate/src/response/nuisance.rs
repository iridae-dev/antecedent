//! Cross-fitted nuisance and outcome-target fits.

use antecedent_stats::{
    AdditiveDesign, FaerBackend, GamWorkspace, GaussianMixtureDensity, PredictorMatrix,
    RoughnessPenalty, SmoothSpec, fit_gam, fit_gam_weighted_design, gaussian_density,
};

use crate::EstimationError;

use super::{
    ADE_NUISANCE_PENALTY, CONDITIONAL_DENSITY_FLOOR, ContinuousResponseEstimator, CrossFitFold, GAM_TARGET_NOT_CONVERGED,
    NUISANCE_BACKFIT, PLUGIN_TARGET_LAMBDA, PseudoOutcome, ensure_crossfit_size, fit_additive,
    fit_additive_design, plugin::predict_one, require_converged_gam, sample::CompleteSample,
    treatment_sigma, treatment_sigma_train_weighted,
};

impl ContinuousResponseEstimator {
    pub(super) fn cross_fitted_pseudo_outcome(
        &self,
        sample: &CompleteSample,
    ) -> Result<PseudoOutcome, EstimationError> {
        let folds = self.cross_fit_folds(sample)?;
        self.cross_fitted_pseudo_outcome_weighted(sample, &folds, None)
    }

    /// The cross-fitting folds of `sample`, with each fold's training designs
    /// expanded once (see [`CrossFitFold`]).
    ///
    /// # Errors
    ///
    /// Too few rows for the folds and basis, or a basis expansion failure.
    pub(super) fn cross_fit_folds(
        &self,
        sample: &CompleteSample,
    ) -> Result<Vec<CrossFitFold>, EstimationError> {
        self.cross_fit_folds_with(sample, RoughnessPenalty::default())
    }

    /// [`Self::cross_fit_folds`] under the average-derivative nuisance penalty.
    pub(super) fn cross_fit_folds_ade(
        &self,
        sample: &CompleteSample,
    ) -> Result<Vec<CrossFitFold>, EstimationError> {
        self.cross_fit_folds_with(sample, ADE_NUISANCE_PENALTY)
    }

    fn cross_fit_folds_with(
        &self,
        sample: &CompleteSample,
        penalty: RoughnessPenalty,
    ) -> Result<Vec<CrossFitFold>, EstimationError> {
        let n = sample.len();
        ensure_crossfit_size(n, self.options.folds, self.options.nuisance_basis)?;
        (0..self.options.folds)
            .map(|fold| {
                let train: Vec<usize> = (0..n).filter(|i| i % self.options.folds != fold).collect();
                let valid: Vec<usize> = (0..n).filter(|i| i % self.options.folds == fold).collect();
                let outcome_design = AdditiveDesign::expand(
                    &sample.raw_subset(&train),
                    train.len(),
                    sample.raw_cols,
                    &self.nuisance_specs(sample.raw_cols, penalty),
                )?;
                let treatment_design = if sample.adjustment_cols == 0 {
                    None
                } else {
                    Some(AdditiveDesign::expand(
                        &sample.adjustment_subset(&train),
                        train.len(),
                        sample.adjustment_cols,
                        &self.nuisance_specs(sample.adjustment_cols, penalty),
                    )?)
                };
                Ok(CrossFitFold {
                    outcome: train.iter().map(|&i| sample.outcome[i]).collect(),
                    treatment: train.iter().map(|&i| sample.treatments[i]).collect(),
                    train,
                    valid,
                    outcome_design,
                    treatment_design,
                })
            })
            .collect()
    }

    /// Penalized nuisance smooths over `ncols` raw columns, knots at the sample
    /// quantiles of the rows they are expanded over.
    fn nuisance_specs(&self, ncols: usize, penalty: RoughnessPenalty) -> Vec<SmoothSpec> {
        (0..ncols)
            .map(|col| {
                SmoothSpec::new(col, self.options.nuisance_basis, self.options.nuisance_lambda)
                    .with_penalty(penalty)
            })
            .collect()
    }

    /// Cross-fitted Kennedy pseudo-outcome, optionally under row weights.
    ///
    /// With `weights`, every fold refits the outcome and treatment nuisances on
    /// its weighted training rows, and the treatment scale, marginal density, and
    /// covariate offset become weighted training-row averages. This is one
    /// Bayesian-bootstrap draw of the whole nuisance stage, not a reweight of the
    /// first-fit pseudo-outcome. `None` is the unweighted estimator (weights of
    /// one, bit-identical). `folds` is [`Self::cross_fit_folds`] of `sample`,
    /// built once by a draw loop and shared by every draw.
    #[allow(
        clippy::too_many_lines,
        reason = "one cross-fitting pass: per fold, fit the outcome and treatment nuisances, then write the row pseudo-outcomes; the loop shares the reused buffers"
    )]
    pub(super) fn cross_fitted_pseudo_outcome_weighted(
        &self,
        sample: &CompleteSample,
        folds: &[CrossFitFold],
        weights: Option<&[f64]>,
    ) -> Result<PseudoOutcome, EstimationError> {
        let n = sample.len();
        ensure_crossfit_size(n, self.options.folds, self.options.nuisance_basis)?;
        if weights.is_some_and(|w| w.len() != n) {
            return Err(EstimationError::stats_msg("pseudo-outcome weights must match rows"));
        }
        let row_weight = |i: usize| weights.map_or(1.0, |w| w[i]);
        let mut pseudo = vec![0.0; n];
        let mut covariate_centered = vec![0.0; n];
        let mut density_floor_rows = 0usize;
        let mut gam_ws = GamWorkspace::default();
        let mut adj_row = vec![0.0; sample.adjustment_cols];
        let mut raw_row = vec![0.0; sample.raw_cols];
        let mut train_weights_buf: Vec<f64> = Vec::new();
        for fold in folds {
            let (train, valid) = (fold.train.as_slice(), fold.valid.as_slice());
            let train_weights: Option<&[f64]> = weights.map(|_| {
                train_weights_buf.clear();
                train_weights_buf.extend(train.iter().map(|&i| row_weight(i)));
                train_weights_buf.as_slice()
            });
            let train_weight_sum = match train_weights {
                Some(w) => w.iter().sum::<f64>(),
                None => train.len() as f64,
            };
            if !train_weight_sum.is_finite() || train_weight_sum <= 0.0 {
                return Err(EstimationError::stats_msg("fold training weights are degenerate"));
            }
            let (outcome_fit, treatment_fit, sigma) =
                Self::fit_train_nuisances(sample, fold, train_weights, &mut gam_ws)?;
            // The training-row treatment means do not depend on the validation row;
            // computing them once per fold avoids |valid| x |train| spline expansions.
            let constant_treatment_mean = match train_weights {
                None => sample.train_treatment_mean(train),
                Some(w) => {
                    train.iter().zip(w).map(|(&i, wi)| wi * sample.treatments[i]).sum::<f64>()
                        / train_weight_sum
                }
            };
            let train_treatment_means: Vec<f64> = match treatment_fit.as_ref() {
                Some(fit) => {
                    let mut means = Vec::with_capacity(train.len());
                    for &j in train {
                        sample.write_adjustment_row(j, &mut adj_row);
                        means.push(predict_one(fit, &adj_row)?);
                    }
                    means
                }
                None => vec![constant_treatment_mean; train.len()],
            };
            // The outcome nuisance is additive, so a counterfactual prediction
            // decomposes exactly: μ̂(a, X_j) = μ̂(A_j, X_j) − f_T(A_j) + f_T(a),
            // where f_T is the centered treatment smooth, giving one smooth
            // evaluation per row instead of a |valid|×|train| full-prediction
            // double loop.
            let treat_smooth = outcome_fit.smooth_for_raw_col(0).ok_or_else(|| {
                EstimationError::unsupported("outcome nuisance is missing its treatment smooth")
            })?;
            // Kennedy et al. (2017, Thm. 3): the marginalization `∫μ̂(a, x) dP_n(x)`
            // is the empirical mean over the *full* sample `P_n`, not the training
            // fold alone — μ̂ is trained out-of-fold, but every row's covariates
            // still contribute to the average that estimates the population
            // marginal (see the identical claim in `mean_curve`'s covariate-term
            // comment). Averaging only the training rows makes the offset the
            // training fold's own covariate mean rather than the population's:
            // unbiased when the held-out rows are an unbiased subsample of every
            // raw column's marginal, which a representative hold-out normally is,
            // but is not guaranteed in general — a fold that happens to omit a
            // covariate's extreme rows shifts its empirical mean away from the
            // population's, biasing every pseudo-outcome the fold produces, and
            // cross-fitting `μ̂` out-of-fold does not fix that shift because the
            // shift is in which rows are averaged, not in which rows trained the
            // model.
            let mut covariate_offset = 0.0;
            let mut covariate_weight_sum = 0.0;
            for (position, &j) in train.iter().enumerate() {
                let treat_partial =
                    outcome_fit.smooth_partial(treat_smooth, sample.treatments[j])?;
                let w = train_weights.map_or(1.0, |tw| tw[position]);
                covariate_offset += w * (outcome_fit.fitted[position] - treat_partial);
                covariate_weight_sum += w;
            }
            for &j in valid {
                sample.write_raw_row(j, &mut raw_row);
                let mu_j = predict_one(&outcome_fit, &raw_row)?;
                let treat_partial =
                    outcome_fit.smooth_partial(treat_smooth, sample.treatments[j])?;
                let w = row_weight(j);
                covariate_offset += w * (mu_j - treat_partial);
                covariate_weight_sum += w;
            }
            if !covariate_weight_sum.is_finite() || covariate_weight_sum <= 0.0 {
                return Err(EstimationError::stats_msg("row weights are degenerate"));
            }
            covariate_offset /= covariate_weight_sum;
            // The marginal treatment density is the (weighted) Gaussian mixture of the
            // training-row means at bandwidth σ; without a treatment model every
            // mean is the same constant and the mixture is one Gaussian.
            let marginal_mixture = if treatment_fit.is_none() {
                GaussianMixtureDensity::new(&[constant_treatment_mean], None, sigma)?
            } else {
                GaussianMixtureDensity::new(&train_treatment_means, train_weights, sigma)?
            };
            for &i in valid {
                sample.write_raw_row(i, &mut raw_row);
                let mu_observed = predict_one(&outcome_fit, &raw_row)?;
                let treatment_mean = match treatment_fit.as_ref() {
                    Some(fit) => {
                        sample.write_adjustment_row(i, &mut adj_row);
                        predict_one(fit, &adj_row)?
                    }
                    None => constant_treatment_mean,
                };
                let raw_density =
                    gaussian_density(sample.treatment_matrix[i], treatment_mean, sigma);
                if !raw_density.is_finite() || raw_density <= CONDITIONAL_DENSITY_FLOOR {
                    density_floor_rows += 1;
                }
                let conditional_density = raw_density.max(CONDITIONAL_DENSITY_FLOOR);
                let marginal_density = marginal_mixture.density(sample.treatment_matrix[i]);
                let marginal_mu = covariate_offset
                    + outcome_fit.smooth_partial(treat_smooth, sample.treatment_matrix[i])?;
                pseudo[i] = marginal_mu
                    + (sample.outcome[i] - mu_observed) * marginal_density / conditional_density;
                covariate_centered[i] = mu_observed - marginal_mu;
            }
        }
        Ok(PseudoOutcome { values: pseudo, density_floor_rows, covariate_centered })
    }

    /// Outcome GAM, treatment GAM (absent without adjusters) and treatment scale
    /// fitted on the fold's training rows, under their row weights `w` when given.
    pub(super) fn fit_train_nuisances(
        sample: &CompleteSample,
        fold: &CrossFitFold,
        w: Option<&[f64]>,
        gam_ws: &mut GamWorkspace,
    ) -> Result<(antecedent_stats::GamFit, Option<antecedent_stats::GamFit>, f64), EstimationError>
    {
        let outcome_fit = fit_additive_design(&fold.outcome_design, &fold.outcome, w, gam_ws)?;
        let treatment_fit = match &fold.treatment_design {
            None => None,
            Some(design) => Some(fit_additive_design(design, &fold.treatment, w, gam_ws)?),
        };
        let sigma = match w {
            None => treatment_sigma(sample, &fold.train, treatment_fit.as_ref())?,
            Some(w) => {
                treatment_sigma_train_weighted(sample, &fold.train, w, treatment_fit.as_ref())?
            }
        };
        Ok((outcome_fit, treatment_fit, sigma))
    }

    /// Unpenalized treatment / penalized adjustment specs for the plug-in target.
    ///
    /// Shared by [`Self::fit_outcome_target`] and the weighted Jacobian path so
    /// the unpenalized-treatment / penalized-adjustment split cannot drift
    /// between the two (ENG-009).
    fn outcome_target_specs(&self, sample: &CompleteSample) -> Vec<SmoothSpec> {
        (0..sample.raw_cols)
            .map(|col| {
                let lambda = if col < sample.treatment_cols {
                    PLUGIN_TARGET_LAMBDA
                } else {
                    self.options.nuisance_lambda
                };
                SmoothSpec::new(col, self.options.nuisance_basis, lambda)
            })
            .collect()
    }

    /// Full-sample outcome GAM whose treatment smooths are the plug-in *target*.
    ///
    /// Same split as [`Self::plugin_gradient_run`]: treatment smooths are
    /// unpenalized cubic regression splines (a roughness penalty with quantile
    /// knots shrinks even a linear dose effect, which biased the g-computation
    /// level by about half its SE at every n in calibration), and
    /// adjustment smooths keep `nuisance_lambda`.
    ///
    /// When the unpenalized target fit fails (a binary or low-cardinality
    /// treatment cannot support an unpenalized cubic basis), the penalized
    /// nuisance fit is used instead and the failure reason is returned so the
    /// caller can disclose that the reported level is the penalized fit.
    pub(super) fn fit_outcome_target(
        &self,
        sample: &CompleteSample,
    ) -> Result<(antecedent_stats::GamFit, Option<String>), EstimationError> {
        let rows: Vec<usize> = (0..sample.len()).collect();
        let x = sample.raw_subset(&rows);
        let specs = self.outcome_target_specs(sample);
        let mut gam_ws = GamWorkspace::default();
        let target = fit_gam(
            &PredictorMatrix::new(&x, sample.len(), sample.raw_cols),
            &sample.outcome,
            &specs,
            &NUISANCE_BACKFIT,
            &FaerBackend,
            &mut gam_ws,
        );
        match target {
            Ok(fit) => Ok((require_converged_gam(fit, GAM_TARGET_NOT_CONVERGED)?, None)),
            // A binary or low-cardinality treatment cannot support an unpenalized
            // cubic basis (singular Gram); keep the penalized nuisance fit there
            // and report why.
            Err(error) => {
                let fit = self.fit_outcome(sample, &rows, &mut gam_ws)?;
                Ok((fit, Some(error.to_string())))
            }
        }
    }

    /// The plug-in target design of [`Self::outcome_target_specs`] expanded once
    /// over every complete row, for the weighted refits of a draw loop.
    pub(super) fn outcome_target_design(
        &self,
        sample: &CompleteSample,
    ) -> Result<AdditiveDesign, EstimationError> {
        let rows: Vec<usize> = (0..sample.len()).collect();
        Ok(AdditiveDesign::expand(
            &sample.raw_subset(&rows),
            sample.len(),
            sample.raw_cols,
            &self.outcome_target_specs(sample),
        )?)
    }

    /// Fit the treatment-target outcome GAM of `design`
    /// ([`Self::outcome_target_design`]) once for this weight vector.
    pub(super) fn fit_outcome_target_weighted(
        sample: &CompleteSample,
        design: &AdditiveDesign,
        weights: Option<&[f64]>,
    ) -> Result<antecedent_stats::GamFit, EstimationError> {
        let mut gam_ws = GamWorkspace::default();
        let fit = fit_gam_weighted_design(
            design,
            &sample.outcome,
            &NUISANCE_BACKFIT,
            weights,
            &mut gam_ws,
        )?;
        if !fit.converged {
            return Err(EstimationError::stats_msg("weighted GAM did not converge"));
        }
        Ok(fit)
    }

    pub(super) fn fit_outcome(
        &self,
        sample: &CompleteSample,
        rows: &[usize],
        workspace: &mut GamWorkspace,
    ) -> Result<antecedent_stats::GamFit, EstimationError> {
        self.fit_outcome_with(sample, rows, RoughnessPenalty::default(), workspace)
    }

    pub(super) fn fit_outcome_with(
        &self,
        sample: &CompleteSample,
        rows: &[usize],
        penalty: RoughnessPenalty,
        workspace: &mut GamWorkspace,
    ) -> Result<antecedent_stats::GamFit, EstimationError> {
        let x = sample.raw_subset(rows);
        let y: Vec<f64> = rows.iter().map(|&i| sample.outcome[i]).collect();
        fit_additive(
            &x,
            rows.len(),
            sample.raw_cols,
            &y,
            self.options.nuisance_basis,
            self.options.nuisance_lambda,
            penalty,
            workspace,
        )
    }

    #[cfg(test)]
    pub(super) fn fit_treatment(
        &self,
        sample: &CompleteSample,
        rows: &[usize],
        workspace: &mut GamWorkspace,
    ) -> Result<Option<antecedent_stats::GamFit>, EstimationError> {
        self.fit_treatment_with(sample, rows, RoughnessPenalty::default(), workspace)
    }

    pub(super) fn fit_treatment_with(
        &self,
        sample: &CompleteSample,
        rows: &[usize],
        penalty: RoughnessPenalty,
        workspace: &mut GamWorkspace,
    ) -> Result<Option<antecedent_stats::GamFit>, EstimationError> {
        if sample.adjustment_cols == 0 {
            return Ok(None);
        }
        let x = sample.adjustment_subset(rows);
        let y: Vec<f64> = rows.iter().map(|&i| sample.treatments[i]).collect();
        fit_additive(
            &x,
            rows.len(),
            sample.adjustment_cols,
            &y,
            self.options.nuisance_basis,
            self.options.nuisance_lambda,
            penalty,
            workspace,
        )
        .map(Some)
    }
}
