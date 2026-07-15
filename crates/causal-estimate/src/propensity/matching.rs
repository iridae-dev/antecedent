//! Propensity-score nearest-neighbor matching.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{AssumptionSet, AverageEffectQuery, ExecutionContext, TargetPopulation};
use causal_data::TabularData;
use causal_expr::IdentifiedEstimand;
use causal_stats::{
    FaerBackend, GlmOptions, MatchingDistance, MatchingIndex, PropensityWorkspace, fit_propensity,
};

use super::prepare::{
    PreparedPropensityProblem, PropensityEstimationWorkspace, PropensityModel, clamp_scores,
    clip_of, default_propensity_overlap, gather, gather_rowmajor, prepare_propensity_problem,
    restrict_to_rows, split_by_treatment, to_row_major, trim_of, trim_retained_rows,
};
use crate::adjustment::EffectEstimate;
use crate::error::EstimationError;
use crate::overlap::{OverlapPolicy, OverlapReport};
use crate::util::{bootstrap_se, sample_std, stats_err, BootstrapSeResult};

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

/// Match each `query` row to its nearest `donor` row; returns `query_y[q] - donor_y[matched]`
/// for every query matched within the caliper (caliper-rejected queries are omitted).
///
/// Reuses [`PropensityEstimationWorkspace`]'s cached [`MatchingIndex`] when donor geometry
/// is unchanged.
pub(crate) fn match_diffs(
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

pub(crate) struct MatchedEstimate {
    pub(crate) ate: f64,
    pub(crate) se_analytic: f64,
}

/// ATT/ATC/ATE via nearest-neighbor matching on `features` (dim columns, row-major).
///
/// ATT matches treated→nearest control; ATC matches control→nearest treated (sign-flipped);
/// ATE pools both directions' per-unit imputed effects (Abadie–Imbens style).
pub(crate) fn matching_contrast(
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
