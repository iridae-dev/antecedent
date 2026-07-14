
/// Inverse-probability weighting estimator (ATE/ATT/ATC via `TargetPopulation`).
///
/// Point estimate is the Hajek (self-normalized) weighted difference of means. Positivity is
/// mandatory: [`OverlapPolicy::ExplicitOverride`] is refused.
#[derive(Clone, Debug)]
pub struct PropensityWeighting {
    /// Dense linear-algebra backend used for the logistic IRLS fit.
    pub backend: FaerBackend,
    /// Bootstrap replicates (0 = skip bootstrap).
    pub bootstrap_replicates: u32,
    /// Overlap policy; must be [`OverlapPolicy::RequireDiagnostics`].
    pub overlap: OverlapPolicy,
    /// GLM fitting options for the propensity model.
    pub glm_options: GlmOptions,
}

impl Default for PropensityWeighting {
    fn default() -> Self {
        Self::new()
    }
}

impl PropensityWeighting {
    /// Defaults: 200 bootstrap replicates, clip = 0.01, no trim.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: FaerBackend,
            bootstrap_replicates: 200,
            overlap: default_propensity_overlap(),
            glm_options: GlmOptions::default(),
        }
    }

    /// Prepare the covariate design from tabular data, identified estimand, and query.
    ///
    /// # Errors
    ///
    /// Overlap policy is `ExplicitOverride`, incompatible estimand, unsupported query, or
    /// missing/invalid data columns.
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedPropensityProblem, EstimationError> {
        prepare_propensity_problem(data, estimand, query, self.overlap)
    }

    /// Fit the propensity model and compute the Hajek-weighted effect, with optional bootstrap.
    ///
    /// # Errors
    ///
    /// Unsupported target population or GLM/backend failure.
    pub fn fit(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut PropensityEstimationWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        let target = IpwTarget::from_population(&problem.target_population)?;
        let trim = trim_of(problem.overlap);
        let model = PropensityModel::fit(
            problem,
            &self.backend,
            &mut workspace.propensity,
            &self.glm_options,
        )?;

        let weights = compute_ipw_weights(
            &problem.treatment,
            &model.clipped_scores,
            &model.fit.scores,
            target,
            trim,
        );
        let ate = hajek_difference(&problem.treatment, &problem.outcome, &weights)?;
        let se_analytic = hajek_analytic_se(&problem.treatment, &problem.outcome, &weights);

        let se_bootstrap = if self.bootstrap_replicates == 0 {
            None
        } else {
            Some(self.bootstrap_se(problem, target, trim, workspace, ctx)?)
        };

        let overlap_report = Some(OverlapReport::from_propensities(
            &model.fit.scores,
            Some(&weights),
            problem.overlap,
        ));

        Ok(EffectEstimate {
            ate,
            se_analytic,
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
        target: IpwTarget,
        trim: Option<f64>,
        workspace: &mut PropensityEstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<f64, EstimationError> {
        let clip = clip_of(problem.overlap);
        let mut rng = ctx.rng.stream(0x9A17_u64);
        let n = problem.nrows;
        let ncols = problem.design_ncols;
        let mut x_boot = vec![0.0; n * ncols];
        let mut t_boot = vec![0.0; n];
        let mut y_boot = vec![0.0; n];
        let mut estimates = Vec::with_capacity(self.bootstrap_replicates as usize);
        for _ in 0..self.bootstrap_replicates {
            for r in 0..n {
                let idx = (rng.next_u64() as usize) % n;
                t_boot[r] = problem.treatment[idx];
                y_boot[r] = problem.outcome[idx];
                for c in 0..ncols {
                    x_boot[c * n + r] = problem.design_matrix[c * n + idx];
                }
            }
            let fit = fit_propensity(
                &x_boot,
                n,
                ncols,
                &t_boot,
                &self.backend,
                &mut workspace.propensity,
                &self.glm_options,
            )
            .map_err(stats_err)?;
            let raw = fit.scores;
            let mut clipped = raw.clone();
            if let Some(c) = clip {
                clamp_scores(&mut clipped, c);
            }
            let w = compute_ipw_weights(&t_boot, &clipped, &raw, target, trim);
            if let Ok(a) = hajek_difference(&t_boot, &y_boot, &w) {
                estimates.push(a);
            }
        }
        if estimates.len() < 2 {
            return Ok(f64::NAN);
        }
        Ok(sample_std(&estimates))
    }
}

// ---------------------------------------------------------------------------------------------
// Stratification
// ---------------------------------------------------------------------------------------------

fn assign_strata(scores: &[f64], n_strata: usize) -> Vec<usize> {
    let n = scores.len();
    let k = n_strata.max(1);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut stratum = vec![0usize; n];
    for (rank, &orig) in order.iter().enumerate() {
        let s = (rank * k) / n.max(1);
        stratum[orig] = s.min(k - 1);
    }
    stratum
}

struct StratifiedResult {
    ate: f64,
    se_analytic: f64,
    /// Fraction of input rows that landed in a stratum with both arms represented. Strata
    /// missing an arm are dropped from the pooled contrast, which redefines the target
    /// population; callers surface this via the overlap report's support figure.
    retained_fraction: f64,
}

fn stratified_ate(
    treatment: &[f64],
    outcome: &[f64],
    stratum: &[usize],
    n_strata: usize,
) -> Result<StratifiedResult, EstimationError> {
    let mut sum1 = vec![0.0; n_strata];
    let mut sq1 = vec![0.0; n_strata];
    let mut cnt1 = vec![0usize; n_strata];
    let mut sum0 = vec![0.0; n_strata];
    let mut sq0 = vec![0.0; n_strata];
    let mut cnt0 = vec![0usize; n_strata];
    for i in 0..treatment.len() {
        let s = stratum[i];
        if treatment[i] > 0.5 {
            sum1[s] += outcome[i];
            sq1[s] += outcome[i] * outcome[i];
            cnt1[s] += 1;
        } else {
            sum0[s] += outcome[i];
            sq0[s] += outcome[i] * outcome[i];
            cnt0[s] += 1;
        }
    }
    let mut diffs = Vec::new();
    let mut ns = Vec::new();
    let mut vars = Vec::new();
    for s in 0..n_strata {
        if cnt1[s] == 0 || cnt0[s] == 0 {
            continue;
        }
        let n1 = cnt1[s] as f64;
        let n0 = cnt0[s] as f64;
        let mean1 = sum1[s] / n1;
        let mean0 = sum0[s] / n0;
        // Unbiased sample variances (÷(n−1)); undefined (NaN) for a single-unit arm, which
        // honestly propagates into `se_analytic` rather than understating it.
        let var1 = sample_variance_from_moments(sq1[s], mean1, cnt1[s]);
        let var0 = sample_variance_from_moments(sq0[s], mean0, cnt0[s]);
        diffs.push(mean1 - mean0);
        ns.push(n1 + n0);
        vars.push(var1 / n1 + var0 / n0);
    }
    let total_n: f64 = ns.iter().sum();
    if total_n <= 0.0 {
        return Err(EstimationError::data_msg("no strata contain both treated and control units"));
    }
    let ate = diffs.iter().zip(&ns).map(|(d, n)| d * n).sum::<f64>() / total_n;
    let se_var = vars.iter().zip(&ns).map(|(v, n)| v * (n / total_n).powi(2)).sum::<f64>();
    let retained_fraction = total_n / (treatment.len().max(1) as f64);
    Ok(StratifiedResult { ate, se_analytic: se_var.sqrt(), retained_fraction })
}

/// Unbiased sample variance from `Σy²`, the mean, and the count (`NaN` when `count < 2`).
fn sample_variance_from_moments(sum_sq: f64, mean: f64, count: usize) -> f64 {
    if count < 2 {
        return f64::NAN;
    }
    let n = count as f64;
    ((sum_sq - n * mean * mean) / (n - 1.0)).max(0.0)
}
