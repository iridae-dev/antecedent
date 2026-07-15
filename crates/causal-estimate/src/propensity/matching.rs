
/// Propensity-score nearest-neighbor matching (Absolute distance, optional caliper).
///
/// Positivity is mandatory: [`OverlapPolicy::ExplicitOverride`] is refused. Supports
/// ATT/ATC/ATE via `TargetPopulation`.
#[derive(Clone, Debug)]
pub struct PropensityMatching {
    /// Dense linear-algebra backend used for the logistic IRLS fit.
    pub backend: FaerBackend,
    /// Bootstrap replicates (0 = skip bootstrap).
    pub bootstrap_replicates: u32,
    /// Overlap policy; must be [`OverlapPolicy::RequireDiagnostics`].
    pub overlap: OverlapPolicy,
    /// GLM fitting options for the propensity model.
    pub glm_options: GlmOptions,
    /// Optional maximum propensity distance for an accepted match.
    pub caliper: Option<f64>,
}

impl Default for PropensityMatching {
    fn default() -> Self {
        Self::new()
    }
}

impl PropensityMatching {
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

    /// Fit the propensity model and compute the matched effect.
    ///
    /// # Errors
    ///
    /// Unsupported target population, empty treated/control arm, no matches within the
    /// caliper, or GLM failure.
    pub fn fit(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut PropensityEstimationWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        let trim = trim_of(problem.overlap);
        let model = PropensityModel::fit(
            problem,
            &self.backend,
            &mut workspace.propensity,
            &self.glm_options,
        )?;
        // Trim on RAW scores (mirrors PropensityWeighting): both query and donor sets are
        // restricted to common-support rows before matching.
        let retained = trim_retained_rows(&model.fit.scores, trim)?;
        let (t_used, y_used, s_used) = restrict_to_rows(
            &problem.treatment,
            &problem.outcome,
            &model.clipped_scores,
            1,
            retained.as_deref(),
        );
        let result = matching_contrast(
            &t_used,
            &y_used,
            &s_used,
            1,
            MatchingDistance::Absolute,
            &problem.target_population,
            self.caliper,
            workspace,
        )?;

        let boot = if self.bootstrap_replicates == 0 {
            None
        } else {
            Some(self.bootstrap_se(problem, trim, workspace, ctx)?)
        };

        let overlap_report =
            Some(OverlapReport::from_propensities(&model.fit.scores, None, problem.overlap));

        Ok(EffectEstimate {
            ate: result.ate,
            se_analytic: result.se_analytic,
            se_bootstrap: None,
            bootstrap_replicates_ok: None,
            bootstrap_replicates_failed: None,
            assumptions,
            overlap: problem.overlap,
            overlap_report,
            retained_memory_bytes: Some(workspace.retained_memory_bytes()),
        }
        .with_bootstrap(boot))
    }

    fn bootstrap_se(
        &self,
        problem: &PreparedPropensityProblem,
        trim: Option<f64>,
        workspace: &mut PropensityEstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<BootstrapSeResult, EstimationError> {
        let clip = clip_of(problem.overlap);
        let mut rng = ctx.rng.stream(0x51E7_u64);
        let n = problem.nrows;
        let ncols = problem.design_ncols;
        let mut x_boot = vec![0.0; n * ncols];
        let mut t_boot = vec![0.0; n];
        let mut y_boot = vec![0.0; n];
        bootstrap_se(self.bootstrap_replicates, &mut rng, n, |idx| {
            for (r, &src) in idx.iter().enumerate() {
                t_boot[r] = problem.treatment[src];
                y_boot[r] = problem.outcome[src];
                for c in 0..ncols {
                    x_boot[c * n + r] = problem.design_matrix[c * n + src];
                }
            }
            let Ok(fit) = fit_propensity(
                &x_boot,
                n,
                ncols,
                &t_boot,
                &self.backend,
                &mut workspace.propensity,
                &self.glm_options,
            ) else {
                return Ok(None);
            };
            let raw = fit.scores;
            let mut scores = raw.clone();
            if let Some(c) = clip {
                clamp_scores(&mut scores, c);
            }
            let Ok(retained) = trim_retained_rows(&raw, trim) else {
                return Ok(None);
            };
            let (t_used, y_used, s_used) =
                restrict_to_rows(&t_boot, &y_boot, &scores, 1, retained.as_deref());
            match matching_contrast(
                &t_used,
                &y_used,
                &s_used,
                1,
                MatchingDistance::Absolute,
                &problem.target_population,
                self.caliper,
                workspace,
            ) {
                Ok(m) => Ok(Some(m.ate)),
                Err(_) => Ok(None),
            }
        })
    }
}
