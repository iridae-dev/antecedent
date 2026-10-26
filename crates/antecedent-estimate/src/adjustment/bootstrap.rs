//! Extracted stage of the parent estimator module.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

impl LinearAdjustmentAte {
    /// Attach bootstrap SE onto a point estimate (progressive uncertainty stage).
    ///
    /// # Errors
    ///
    /// Bootstrap failure.
    pub fn attach_bootstrap(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
        point: EffectEstimate,
    ) -> Result<EffectEstimate, EstimationError> {
        let boot = if self.inference.bootstrap_replicates == 0 {
            None
        } else {
            let t_col = problem
                .design
                .treatment_column()
                .ok_or_else(|| EstimationError::stats_msg("missing treatment column"))?;
            Some(self.bootstrap_se(problem, workspace, ctx, t_col)?)
        };
        Ok(match boot {
            Some(outcome) => point.with_bootstrap_outcome(outcome),
            None => point.with_bootstrap(None),
        })
    }

    /// ATE from the configured linear fit on a row-resample of a prepared design.
    ///
    /// `row_src[r]` copies design row `row_src[r]` onto output row `r`. Length must
    /// equal `problem.design.nrows`. For `AllObserved` / `Predicate` the ATE is
    /// `β_T · Δ` and does not reread treatment labels.
    ///
    /// # Errors
    ///
    /// Shape mismatch, out-of-range source row, or fit failure.
    pub fn ate_on_row_indices(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        row_src: &[usize],
    ) -> Result<f64, EstimationError> {
        let n = problem.design.nrows;
        let p = problem.design.ncols;
        if row_src.len() != n {
            return Err(EstimationError::data_msg(
                "row resample length must match the prepared design",
            ));
        }
        let mut x_boot = vec![0.0; n * p];
        let mut y_boot = vec![0.0; n];
        self.ate_on_row_indices_into(problem, workspace, row_src, &mut x_boot, &mut y_boot)
    }

    /// [`Self::ate_on_row_indices`] using caller-owned gather buffers.
    ///
    /// Here `row_src` may be shorter than the design (a sub-window resample); the
    /// buffers must hold `row_src.len()` rows (`x_boot`: `row_src.len() · ncols`).
    ///
    /// # Errors
    ///
    /// Shape mismatch, out-of-range source row, or fit failure.
    pub fn ate_on_row_indices_into(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        row_src: &[usize],
        x_boot: &mut [f64],
        y_boot: &mut [f64],
    ) -> Result<f64, EstimationError> {
        let t_col = problem
            .design
            .treatment_column()
            .ok_or_else(|| EstimationError::stats_msg("missing treatment column"))?;
        let coefs = self.fit_resampled_coefficients(problem, workspace, row_src, x_boot, y_boot)?;
        gcomp_or_coef_ate(problem, &coefs, t_col)
    }

    pub(super) fn fit_resampled_coefficients(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        row_src: &[usize],
        x_boot: &mut [f64],
        y_boot: &mut [f64],
    ) -> Result<Vec<f64>, EstimationError> {
        let rows = problem.design.nrows;
        let p = problem.design.ncols;
        // Output rows follow the row map, which may be a sub-window of the design
        // (a multi-design shared time window); every source row must exist.
        let n = row_src.len();
        if y_boot.len() != n || x_boot.len() != n * p {
            return Err(EstimationError::data_msg("row resample buffers must match the row map"));
        }
        for (r, &src) in row_src.iter().enumerate() {
            if src >= rows {
                return Err(EstimationError::data_msg("row resample index out of range"));
            }
            y_boot[r] = problem.design.outcome[src];
            for c in 0..p {
                x_boot[c * n + r] = problem.design.matrix[c * rows + src];
            }
        }
        if self.fit_kind == LinearFitKind::Ols {
            return self
                .backend
                .least_squares_coefficients(x_boot, n, p, y_boot)
                .map_err(EstimationError::from);
        }
        Ok(solve_linear(self.fit_kind, self.backend, x_boot, n, p, y_boot, &mut workspace.ols)?
            .coefficients)
    }

    pub(super) fn bootstrap_clusters(
        &self,
        problem: &PreparedEstimationProblem,
    ) -> Result<Option<std::borrow::Cow<'_, [u32]>>, EstimationError> {
        let labels = crate::se::declared_row_labels(
            self.inference.cluster_ids.as_deref(),
            self.inference.source_cluster_ids.as_deref(),
            problem.design.row_selection.iter().copied(),
        )?;
        if let Some(labels) = &labels {
            crate::se::require_clusters(Some(labels.as_ref()), problem.design.nrows)?;
        }
        Ok(labels)
    }

    pub(super) fn series_block(&self, n: usize) -> Option<usize> {
        self.inference.panel_times.as_ref().map(|_| {
            let span = match self.inference.se_kind {
                AnalyticSeKind::NeweyWest { lag } | AnalyticSeKind::PanelClusterHac { lag } => {
                    lag.max(1)
                }
                _ => 1,
            };
            antecedent_data::circular_block_length(span, n)
        })
    }

    pub(super) fn bootstrap_se(
        &self,
        problem: &PreparedEstimationProblem,
        _workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
        t_col: usize,
    ) -> Result<crate::util::BootstrapOutcome, EstimationError> {
        let n = problem.design.nrows;
        let p = problem.design.ncols;
        // Replicates own a cold workspace. The caller's scratch stays untouched: parallel
        // workers cannot share it, and a warm IRLS start would change the replicate fit.
        let resampling = (self.fit_kind == LinearFitKind::Ols
            && self.inference.bootstrap_replicates >= 32)
            .then(|| {
                antecedent_stats::faer_backend::ResampledLeastSquares::prepare(
                    &problem.design.matrix,
                    n,
                    p,
                    &problem.design.outcome,
                    ctx,
                )
            })
            .flatten();
        let clusters = self.bootstrap_clusters(problem)?;
        let series_block = self.series_block(n);
        // Each worker gathers `n * p` design cells and `n` outcomes.
        let _gather =
            crate::util::admit_gather_scratch(ctx, n, p + 1, self.inference.bootstrap_replicates)?;
        crate::util::bootstrap_outcome_by_design(
            self.inference.bootstrap_interval.as_ref(),
            crate::util::BootstrapDesign {
                clusters: clusters.as_deref(),
                series_block,
                declared: self.inference.sampling_design,
            },
            self.inference.bootstrap_replicates,
            ctx,
            0xA7E_u64,
            n,
            || {
                (
                    EstimationWorkspace::default(),
                    Vec::new(),
                    Vec::new(),
                    antecedent_stats::faer_backend::ResampledLeastSquaresWorkspace::default(),
                )
            },
            |(ws, x_boot, y_boot, summary_ws), idx| {
                let same_length = idx.len() == n;
                let candidate = if same_length {
                    match &resampling {
                        Some(prepared) => prepared.try_coefficients(idx, summary_ws),
                        None => Ok(None),
                    }
                } else {
                    Ok(None)
                };
                let coefficients = match candidate {
                    Ok(Some(coefs)) => Ok(coefs),
                    Ok(None) => {
                        let m = idx.len();
                        x_boot.resize(m * p, 0.0);
                        y_boot.resize(m, 0.0);
                        self.fit_resampled_coefficients(problem, ws, idx, x_boot, y_boot)
                    }
                    Err(error) => Err(EstimationError::from(error)),
                };
                match coefficients {
                    Ok(coefs) => Ok(Some(gcomp_or_coef_ate(problem, &coefs, t_col)?)),
                    Err(error) if bootstrap_fit_failed(&error) => Ok(None),
                    Err(error) => Err(error),
                }
            },
        )
    }
}
