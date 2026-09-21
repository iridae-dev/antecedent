//! Covariate-distance matching estimator.
//!
//! Analytic standard errors follow Abadie–Imbens (2006) via
//! [`super::matching::matching_contrast`]. The nonparametric bootstrap is invalid for
//! nearest-neighbor matching with a fixed number of matches (Abadie–Imbens 2008); the
//! licensed uncertainty product is [`EffectEstimate::se_analytic`]. Setting
//! [`DistanceMatching::bootstrap_replicates`] does **not** populate
//! [`EffectEstimate::se_bootstrap`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{AssumptionSet, AverageEffectQuery, ExecutionContext, PopulationRegistry};
use antecedent_data::TabularData;
use antecedent_expr::IdentifiedEstimand;
use antecedent_stats::{FaerBackend, GlmOptions, MatchingDistance};

use super::matching::matching_contrast;
use super::prepare::{
    PreparedPropensityProblem, PropensityEstimationWorkspace, PropensityModel,
    default_propensity_overlap, gather_optional_multiway, gather_optional_row_labels,
    prepare_propensity_problem_with_registry, restrict_to_rows, to_row_major, trim_of,
    trim_retained_rows,
};
use crate::adjustment::EffectEstimate;
use crate::error::EstimationError;
use crate::overlap::{IpwTarget, OverlapPolicy};
use crate::se::AnalyticSeKind;

/// Distance matching on z-scored adjustment covariates (Euclidean), not the propensity score.
///
/// Features are standardized (mean 0, sd 1) on the estimation sample so the metric is not
/// unit-dependent. A caliper, when set, is in that standardized Euclidean metric.
///
/// A propensity model is still fit to populate mandatory positivity diagnostics
/// ([`EffectEstimate::overlap_report`]) and — when the overlap policy sets a trim threshold —
/// to restrict matching to common-support rows; it does not otherwise influence the
/// covariate-space matched contrast. Positivity is mandatory:
/// [`OverlapPolicy::ExplicitOverride`] is refused.
///
/// Analytic SEs use Abadie–Imbens (2006) donor-reuse variance; see module docs for the
/// bootstrap caveat (Abadie–Imbens 2008).
#[derive(Clone, Debug)]
pub struct DistanceMatching {
    /// Dense linear-algebra backend used for the diagnostic logistic fit.
    pub backend: FaerBackend,
    /// Accepted for API compatibility; does not populate [`EffectEstimate::se_bootstrap`]
    /// (Abadie–Imbens 2008 — see module docs). Prefer [`EffectEstimate::se_analytic`].
    pub bootstrap_replicates: u32,
    /// Overlap policy; must be [`OverlapPolicy::RequireDiagnostics`].
    pub overlap: OverlapPolicy,
    /// GLM fitting options for the diagnostic propensity model.
    pub glm_options: GlmOptions,
    /// Optional maximum Euclidean distance for an accepted match.
    pub caliper: Option<f64>,
    /// Analytic SE kind.
    pub se_kind: AnalyticSeKind,
    /// Optional cluster ids aligned to prepared complete-case rows.
    pub cluster_ids: Option<Vec<u32>>,
    /// Optional bindings for named predicates / custom target distributions.
    pub population_registry: Option<PopulationRegistry>,
    /// Multiway cluster ids (one `Vec<u32>` per clustering dimension).
    pub multiway_ids: Option<Vec<Vec<u32>>>,
    /// Optional panel time labels for panel HAC.
    pub panel_times: Option<Vec<i64>>,
}

impl Default for DistanceMatching {
    fn default() -> Self {
        Self::new()
    }
}

impl DistanceMatching {
    /// Defaults: no caliper, `bootstrap_replicates = 200` (ignored for SE), clip = 0.01, no trim.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: FaerBackend,
            bootstrap_replicates: 200,
            overlap: default_propensity_overlap(),
            glm_options: GlmOptions::default(),
            caliper: None,
            se_kind: AnalyticSeKind::Homoskedastic,
            cluster_ids: None,
            population_registry: None,
            multiway_ids: None,
            panel_times: None,
        }
    }

    /// Set the dense linear-algebra backend used for the diagnostic logistic fit.
    #[must_use]
    pub const fn with_backend(mut self, backend: FaerBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Record a bootstrap-replicate count without licensing a bootstrap SE.
    ///
    /// The nonparametric bootstrap is invalid for nearest-neighbor matching CIs
    /// (Abadie–Imbens 2008). This setter keeps the field for callers that still set it;
    /// [`DistanceMatching::fit`] never writes [`EffectEstimate::se_bootstrap`]. Prefer the
    /// analytic SE.
    #[must_use]
    pub const fn with_bootstrap_replicates(mut self, replicates: u32) -> Self {
        self.bootstrap_replicates = replicates;
        self
    }

    /// Set the overlap policy. Positivity is mandatory here:
    /// [`OverlapPolicy::ExplicitOverride`] is refused by `prepare`.
    #[must_use]
    pub const fn with_overlap(mut self, overlap: OverlapPolicy) -> Self {
        self.overlap = overlap;
        self
    }

    /// Set the GLM fitting options for the diagnostic propensity model.
    #[must_use]
    pub const fn with_glm_options(mut self, glm_options: GlmOptions) -> Self {
        self.glm_options = glm_options;
        self
    }

    /// Set the maximum Euclidean distance for an accepted match.
    ///
    /// Defaults to `None` (no caliper): every query row is matched to its nearest donor
    /// regardless of distance.
    #[must_use]
    pub const fn with_caliper(mut self, caliper: f64) -> Self {
        self.caliper = Some(caliper);
        self
    }

    /// Set the analytic SE kind.
    #[must_use]
    pub const fn with_se_kind(mut self, se_kind: AnalyticSeKind) -> Self {
        self.se_kind = se_kind;
        self
    }

    /// Set cluster ids aligned to prepared complete-case rows.
    #[must_use]
    pub fn with_cluster_ids(mut self, cluster_ids: Vec<u32>) -> Self {
        self.cluster_ids = Some(cluster_ids);
        self
    }

    /// Set bindings for named predicates / custom target distributions.
    #[must_use]
    pub fn with_population_registry(mut self, registry: PopulationRegistry) -> Self {
        self.population_registry = Some(registry);
        self
    }

    /// Set multiway cluster ids (one `Vec<u32>` per clustering dimension).
    #[must_use]
    pub fn with_multiway_ids(mut self, multiway_ids: Vec<Vec<u32>>) -> Self {
        self.multiway_ids = Some(multiway_ids);
        self
    }

    /// Set panel time labels for panel HAC.
    #[must_use]
    pub fn with_panel_times(mut self, panel_times: Vec<i64>) -> Self {
        self.panel_times = Some(panel_times);
        self
    }

    /// Prepare the covariate design.
    ///
    /// # Errors
    ///
    /// See [`PropensityWeighting::prepare`](crate::propensity::PropensityWeighting::prepare).
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedPropensityProblem, EstimationError> {
        prepare_propensity_problem_with_registry(
            data,
            estimand,
            query,
            self.overlap,
            self.population_registry.as_ref(),
        )
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
        _ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        // `bootstrap_replicates` is intentionally unread: the nonparametric bootstrap is
        // invalid for fixed-M NN matching (Abadie–Imbens 2008) and must not land in
        // `se_bootstrap` even when a caller still requests replicates.
        let _ = self.bootstrap_replicates;
        if problem.adjustment_set.is_empty() {
            return Err(EstimationError::unsupported(
                "distance matching requires a non-empty adjustment set",
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
        let (t_used, y_used, mut f_used) = restrict_to_rows(
            &problem.treatment,
            &problem.outcome,
            &features,
            dim,
            retained.as_deref(),
        );
        standardize_rowmajor_inplace(&mut f_used, t_used.len(), dim);
        let tw_used: Option<Vec<f64>> = problem.target_weights.as_ref().map(|w| match &retained {
            Some(idx) => idx.iter().map(|&i| w[i]).collect(),
            None => w.to_vec(),
        });
        let clusters_used = gather_optional_row_labels(
            self.cluster_ids.as_deref(),
            problem.nrows,
            retained.as_deref(),
            "cluster_ids",
        )?;
        let times_used = gather_optional_row_labels(
            self.panel_times.as_deref(),
            problem.nrows,
            retained.as_deref(),
            "panel_times",
        )?;
        let multiway_used = gather_optional_multiway(
            self.multiway_ids.as_deref(),
            problem.nrows,
            retained.as_deref(),
        )?;
        let result = matching_contrast(
            &t_used,
            &y_used,
            &f_used,
            dim,
            MatchingDistance::Euclidean,
            &problem.target_population,
            self.caliper,
            workspace,
            self.se_kind,
            clusters_used.as_deref(),
            tw_used.as_deref(),
            multiway_used.as_ref(),
            times_used.as_deref(),
        )?;

        let ipw_target = IpwTarget::from_population(&problem.target_population).ok();
        let mut overlap_report = crate::propensity::propensity_overlap_report(
            problem,
            &diag.fit.scores,
            None,
            ipw_target,
        );
        overlap_report.retained_fraction *= result.retained_fraction;
        let overlap_report = Some(overlap_report);

        Ok(EffectEstimate::new(result.ate, result.se_analytic, assumptions, problem.overlap)
            .with_se_kind(self.se_kind)
            .with_n_obs(u64::try_from(result.n_obs).unwrap_or(u64::MAX))
            .with_overlap_report(overlap_report)
            .with_retained_memory_bytes(Some(workspace.retained_memory_bytes())))
    }
}

/// Z-score each feature column on the current sample (row-major `n × dim`).
fn standardize_rowmajor_inplace(features: &mut [f64], n: usize, dim: usize) {
    if n == 0 || dim == 0 {
        return;
    }
    let nf = n as f64;
    for c in 0..dim {
        let mut mean = 0.0;
        for r in 0..n {
            mean += features[r * dim + c];
        }
        mean /= nf;
        let mut var = 0.0;
        for r in 0..n {
            let d = features[r * dim + c] - mean;
            var += d * d;
        }
        let sd = (var / nf).sqrt().max(1e-12);
        for r in 0..n {
            features[r * dim + c] = (features[r * dim + c] - mean) / sd;
        }
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::StreamDomain;

    use std::sync::Arc;

    use antecedent_core::{
        AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, MeasurementSpec, RoleHint,
        SmallRoleSet, TargetPopulation, ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use antecedent_expr::{ExprId, IdentifiedEstimand};
    use antecedent_kernels::standard_normal;

    use super::*;

    fn confounded_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
        let mut rng =
            ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0x1234_u64);
        let mut z = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let zi = standard_normal(&mut rng);
            let logit = -0.5 + zi;
            let p = 1.0 / (1.0 + (-logit).exp());
            let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
            let noise = standard_normal(&mut rng) * 0.5;
            z[i] = zi;
            t[i] = ti;
            y[i] = 2.0 * ti + zi + noise;
        }
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "t",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "z",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(z),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([VariableId::from_raw(2)]),
            ExprId::from_raw(0),
        );
        (TabularData::new(storage), estimand)
    }

    /// Regression: requesting bootstrap replicates must not license an SE in
    /// `se_bootstrap` (Abadie–Imbens 2008). Analytic SE remains the uncertainty product.
    #[test]
    fn bootstrap_replicates_do_not_populate_se_bootstrap() {
        let (data, estimand) = confounded_scm(120, 43);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::Treated);
        let est = DistanceMatching { bootstrap_replicates: 30, ..DistanceMatching::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = PropensityEstimationWorkspace::default();
        let effect =
            est.fit(&prep, &mut ws, &ExecutionContext::for_tests(7), AssumptionSet::new()).unwrap();
        assert!(
            effect.se_bootstrap.is_none(),
            "NN matching must not store the invalid bootstrap SE"
        );
        assert!(
            effect.se_analytic.is_finite() && effect.se_analytic > 0.0,
            "se_analytic={}",
            effect.se_analytic
        );
    }
}
