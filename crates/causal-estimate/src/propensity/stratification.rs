
/// Propensity stratification estimator: within-stratum difference of means pooled by size.
///
/// Supports [`TargetPopulation::AllObserved`] only (stratified ATE). Positivity is mandatory:
/// [`OverlapPolicy::ExplicitOverride`] is refused.
#[derive(Clone, Debug)]
pub struct PropensityStratification {
    /// Dense linear-algebra backend used for the logistic IRLS fit.
    pub backend: FaerBackend,
    /// Bootstrap replicates (0 = skip bootstrap).
    pub bootstrap_replicates: u32,
    /// Overlap policy; must be [`OverlapPolicy::RequireDiagnostics`].
    pub overlap: OverlapPolicy,
    /// GLM fitting options for the propensity model.
    pub glm_options: GlmOptions,
    /// Number of quantile strata (default 5).
    pub n_strata: u32,
}

impl Default for PropensityStratification {
    fn default() -> Self {
        Self::new()
    }
}

impl PropensityStratification {
    /// Defaults: 5 strata, 200 bootstrap replicates, clip = 0.01, no trim.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: FaerBackend,
            bootstrap_replicates: 200,
            overlap: default_propensity_overlap(),
            glm_options: GlmOptions::default(),
            n_strata: 5,
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

    /// Fit the propensity model, assign quantile strata, and compute the pooled stratified ATE.
    ///
    /// # Errors
    ///
    /// Unsupported target population, no strata with both arms represented, or GLM failure.
    pub fn fit(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut PropensityEstimationWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        if !matches!(problem.target_population, TargetPopulation::AllObserved) {
            return Err(EstimationError::UnsupportedQuery(
                "propensity stratification only supports TargetPopulation::AllObserved".into(),
            ));
        }
        let trim = trim_of(problem.overlap);
        let model = PropensityModel::fit(
            problem,
            &self.backend,
            &mut workspace.propensity,
            &self.glm_options,
        )?;
        let n_strata = (self.n_strata.max(1)) as usize;
        // Trim on RAW scores (mirrors PropensityWeighting): stratify only common-support rows.
        let retained = trim_retained_rows(&model.fit.scores, trim)?;
        let (t_used, y_used, s_used) = restrict_to_rows(
            &problem.treatment,
            &problem.outcome,
            &model.clipped_scores,
            1,
            retained.as_deref(),
        );
        let stratum = assign_strata(&s_used, n_strata);
        let result = stratified_ate(&t_used, &y_used, &stratum, n_strata)?;

        let boot = if self.bootstrap_replicates == 0 {
            None
        } else {
            Some(self.bootstrap_se(problem, n_strata, trim, workspace, ctx)?)
        };

        let mut report = OverlapReport::from_propensities(&model.fit.scores, None, problem.overlap);
        // Strata missing a treatment arm are dropped from the pooled contrast; fold the
        // retained fraction into the support figure so the artifact reflects the population
        // the estimate actually targets.
        report.target_population_support *= result.retained_fraction;
        let overlap_report = Some(report);

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
        n_strata: usize,
        trim: Option<f64>,
        workspace: &mut PropensityEstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<BootstrapSeResult, EstimationError> {
        let clip = clip_of(problem.overlap);
        let mut rng = ctx.rng.stream(0x3D2F_u64);
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
            let stratum = assign_strata(&s_used, n_strata);
            match stratified_ate(&t_used, &y_used, &stratum, n_strata) {
                Ok(r) => Ok(Some(r.ate)),
                Err(_) => Ok(None),
            }
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Matching (shared by PropensityMatching and DistanceMatching)
// ---------------------------------------------------------------------------------------------

pub(crate) fn split_by_treatment(treatment: &[f64]) -> (Vec<usize>, Vec<usize>) {
    let mut treated = Vec::new();
    let mut control = Vec::new();
    for (i, &t) in treatment.iter().enumerate() {
        if t > 0.5 {
            treated.push(i);
        } else {
            control.push(i);
        }
    }
    (treated, control)
}

pub(crate) fn gather(values: &[f64], idx: &[usize]) -> Vec<f64> {
    idx.iter().map(|&i| values[i]).collect()
}

fn gather_rowmajor(matrix: &[f64], dim: usize, idx: &[usize]) -> Vec<f64> {
    let mut out = Vec::with_capacity(idx.len() * dim);
    for &i in idx {
        out.extend_from_slice(&matrix[i * dim..(i + 1) * dim]);
    }
    out
}

/// Match each `query` row to its nearest `donor` row; returns `query_y[q] - donor_y[matched]`
/// for every query matched within the caliper (caliper-rejected queries are omitted).
///
/// Reuses [`PropensityEstimationWorkspace`]'s cached [`MatchingIndex`] when donor geometry
/// is unchanged.
fn match_diffs(
    donor_features: &[f64],
    donor_outcome: &[f64],
    dim: usize,
    distance: MatchingDistance,
    query_features: &[f64],
    query_outcome: &[f64],
    caliper: Option<f64>,
    workspace: &mut PropensityEstimationWorkspace,
) -> Result<Vec<f64>, EstimationError> {
    let n_donors = donor_outcome.len();
    if n_donors == 0 {
        return Err(EstimationError::data_msg("matching requires at least one donor row"));
    }
    workspace.ensure_matching_index(donor_features, dim, distance)?;
    let n_queries = query_outcome.len();
    let mut donor_rows = std::mem::take(&mut workspace.matching_donor_rows);
    let mut distances = std::mem::take(&mut workspace.matching_distances);
    donor_rows.clear();
    donor_rows.resize(n_queries, 0);
    distances.clear();
    distances.resize(n_queries, 0.0);
    {
        let index = workspace.matching_index.as_ref().expect("ensured");
        index
            .match_all(query_features, n_queries, caliper, &mut donor_rows, &mut distances)
            .map_err(stats_err)?;
    }
    let mut diffs = Vec::with_capacity(n_queries);
    for q in 0..n_queries {
        let d = donor_rows[q];
        if d != usize::MAX {
            diffs.push(query_outcome[q] - donor_outcome[d]);
        }
    }
    workspace.matching_donor_rows = donor_rows;
    workspace.matching_distances = distances;
    Ok(diffs)
}

struct MatchedEstimate {
    ate: f64,
    se_analytic: f64,
}

/// ATT/ATC/ATE via nearest-neighbor matching on `features` (dim columns, row-major).
///
/// ATT matches treated→nearest control; ATC matches control→nearest treated (sign-flipped);
/// ATE pools both directions' per-unit imputed effects (Abadie–Imbens style).
fn matching_contrast(
    treatment: &[f64],
    outcome: &[f64],
    features: &[f64],
    dim: usize,
    distance: MatchingDistance,
    target: &TargetPopulation,
    caliper: Option<f64>,
    workspace: &mut PropensityEstimationWorkspace,
) -> Result<MatchedEstimate, EstimationError> {
    let (treated_idx, control_idx) = split_by_treatment(treatment);
    if treated_idx.is_empty() || control_idx.is_empty() {
        return Err(EstimationError::data_msg("matching requires both treated and control rows"));
    }
    let treated_feat = gather_rowmajor(features, dim, &treated_idx);
    let control_feat = gather_rowmajor(features, dim, &control_idx);
    let treated_y = gather(outcome, &treated_idx);
    let control_y = gather(outcome, &control_idx);

    let per_unit_effects: Vec<f64> = match target {
        TargetPopulation::Treated => match_diffs(
            &control_feat,
            &control_y,
            dim,
            distance,
            &treated_feat,
            &treated_y,
            caliper,
            workspace,
        )?,
        TargetPopulation::Untreated => match_diffs(
            &treated_feat,
            &treated_y,
            dim,
            distance,
            &control_feat,
            &control_y,
            caliper,
            workspace,
        )?
        .into_iter()
        .map(|d| -d)
        .collect(),
        TargetPopulation::AllObserved => {
            let mut att_diffs = match_diffs(
                &control_feat,
                &control_y,
                dim,
                distance,
                &treated_feat,
                &treated_y,
                caliper,
                workspace,
            )?;
            let atc_diffs: Vec<f64> = match_diffs(
                &treated_feat,
                &treated_y,
                dim,
                distance,
                &control_feat,
                &control_y,
                caliper,
                workspace,
            )?
            .into_iter()
            .map(|d| -d)
            .collect();
            att_diffs.extend(atc_diffs);
            att_diffs
        }
        _ => {
            return Err(EstimationError::UnsupportedQuery(
                "matching estimators support AllObserved, Treated, or Untreated target populations"
                    .into(),
            ));
        }
    };
    if per_unit_effects.is_empty() {
        return Err(EstimationError::data_msg("no matched units within caliper"));
    }
    let n = per_unit_effects.len() as f64;
    let ate = per_unit_effects.iter().sum::<f64>() / n;
    let se_analytic = sample_std(&per_unit_effects) / n.sqrt();
    Ok(MatchedEstimate { ate, se_analytic })
}
