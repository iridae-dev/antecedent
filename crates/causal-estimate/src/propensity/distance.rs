
/// Distance matching on raw adjustment covariates (Euclidean), not the propensity score.
///
/// A propensity model is still fit to populate mandatory positivity diagnostics
/// ([`EffectEstimate::overlap_report`]) and — when the overlap policy sets a trim threshold —
/// to restrict matching to common-support rows; it does not otherwise influence the
/// covariate-space matched contrast. Positivity is mandatory:
/// [`OverlapPolicy::ExplicitOverride`] is refused.
#[derive(Clone, Debug)]
pub struct DistanceMatching {
    /// Dense linear-algebra backend used for the diagnostic logistic fit.
    pub backend: FaerBackend,
    /// Bootstrap replicates (0 = skip bootstrap).
    pub bootstrap_replicates: u32,
    /// Overlap policy; must be [`OverlapPolicy::RequireDiagnostics`].
    pub overlap: OverlapPolicy,
    /// GLM fitting options for the diagnostic propensity model.
    pub glm_options: GlmOptions,
    /// Optional maximum Euclidean distance for an accepted match.
    pub caliper: Option<f64>,
}

impl Default for DistanceMatching {
    fn default() -> Self {
        Self::new()
    }
}

impl DistanceMatching {
    /// Defaults: no caliper, 200 bootstrap replicates, clip = 0.01, no trim.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: FaerBackend,
            bootstrap_replicates: 200,
            overlap: default_propensity_overlap(),
            glm_options: GlmOptions::default(),
            caliper: None,
        }
    }

    /// Prepare the covariate design.
    ///
    /// # Errors
    ///
    /// See [`PropensityWeighting::prepare`].
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedPropensityProblem, EstimationError> {
        prepare_propensity_problem(data, estimand, query, self.overlap)
    }

    /// Match on raw covariates (Euclidean) and compute the matched effect; fits a diagnostic
    /// propensity model for the mandatory overlap report.
    ///
    /// # Errors
    ///
    /// Empty adjustment set, unsupported target population, empty treated/control arm, no
    /// matches within the caliper, or GLM failure (diagnostic fit).
    pub fn fit(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut PropensityEstimationWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        if problem.adjustment_set.is_empty() {
            return Err(EstimationError::UnsupportedQuery(
                "distance matching requires a non-empty adjustment set".into(),
            ));
        }
        let trim = trim_of(problem.overlap);
        let dim = problem.covariates.len();
        let features = to_row_major(&problem.covariates, problem.nrows);

        // Diagnostic propensity fit: populates the mandatory overlap report and, when a trim
        // threshold is configured, restricts both query and donor sets to the common-support
        // rows (raw scores) — it does not otherwise influence the covariate-space contrast.
        let diag = PropensityModel::fit(
            problem,
            &self.backend,
            &mut workspace.propensity,
            &self.glm_options,
        )?;
        let retained = trim_retained_rows(&diag.fit.scores, trim)?;
        let (t_used, y_used, f_used) = restrict_to_rows(
            &problem.treatment,
            &problem.outcome,
            &features,
            dim,
            retained.as_deref(),
        );
        let result = matching_contrast(
            &t_used,
            &y_used,
            &f_used,
            dim,
            MatchingDistance::Euclidean,
            &problem.target_population,
            self.caliper,
            workspace,
        )?;

        let se_bootstrap = if self.bootstrap_replicates == 0 {
            None
        } else {
            Some(self.bootstrap_se(problem, dim, &features, trim, workspace, ctx))
        };

        let overlap_report =
            Some(OverlapReport::from_propensities(&diag.fit.scores, None, problem.overlap));

        Ok(EffectEstimate {
            ate: result.ate,
            se_analytic: result.se_analytic,
            se_bootstrap,
            assumptions,
            overlap: problem.overlap,
            overlap_report,
            retained_memory_bytes: Some(workspace.retained_memory_bytes()),
        })
    }

    fn bootstrap_se(
        &self,
        problem: &PreparedPropensityProblem,
        dim: usize,
        features: &[f64],
        trim: Option<f64>,
        workspace: &mut PropensityEstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> f64 {
        let mut rng = ctx.rng.stream(0x7C11_u64);
        let n = problem.nrows;
        let ncols = problem.design_ncols;
        let mut feat_boot = vec![0.0; n * dim];
        // Diagnostic design resample, needed only to recompute the trim per replicate.
        let mut x_boot = if trim.is_some() { vec![0.0; n * ncols] } else { Vec::new() };
        let mut t_boot = vec![0.0; n];
        let mut y_boot = vec![0.0; n];
        let mut estimates = Vec::with_capacity(self.bootstrap_replicates as usize);
        for _ in 0..self.bootstrap_replicates {
            for r in 0..n {
                let idx = (rng.next_u64() as usize) % n;
                t_boot[r] = problem.treatment[idx];
                y_boot[r] = problem.outcome[idx];
                for d in 0..dim {
                    feat_boot[r * dim + d] = features[idx * dim + d];
                }
                if trim.is_some() {
                    for c in 0..ncols {
                        x_boot[c * n + r] = problem.design_matrix[c * n + idx];
                    }
                }
            }
            let retained = if trim.is_some() {
                let Ok(fit) = fit_propensity(
                    &x_boot,
                    n,
                    ncols,
                    &t_boot,
                    &self.backend,
                    &mut workspace.propensity,
                    &self.glm_options,
                ) else {
                    continue;
                };
                match trim_retained_rows(&fit.scores, trim) {
                    Ok(r) => r,
                    Err(_) => continue,
                }
            } else {
                None
            };
            let (t_used, y_used, f_used) =
                restrict_to_rows(&t_boot, &y_boot, &feat_boot, dim, retained.as_deref());
            if let Ok(m) = matching_contrast(
                &t_used,
                &y_used,
                &f_used,
                dim,
                MatchingDistance::Euclidean,
                &problem.target_population,
                self.caliper,
                workspace,
            ) {
                estimates.push(m.ate);
            }
        }
        if estimates.len() < 2 {
            return f64::NAN;
        }
        sample_std(&estimates)
    }
}
