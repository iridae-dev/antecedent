//! Observational contrasts between levels of a multi-level treatment.
//!
//! A treatment with more than two levels has no single "the" average effect, so the target
//! population is chosen explicitly and reported on the result, never inferred:
//!
//! * [`MultiLevelTarget::PairwiseSubpopulation`] contrasts the declared active and control
//!   levels on the rows whose treatment is at one of the two (`T in {a, b}`). Rows at other
//!   levels are outside the population the contrast speaks for. This is binary AIPW on that
//!   subpopulation.
//! * [`MultiLevelTarget::FullPopulation`] standardizes over the whole complete-case
//!   population with a generalized propensity score: a multinomial logit `e_k(Z) =
//!   P(T = k | Z)` over the declared level set and a per-level outcome regression
//!   `m_k(Z) = E[Y | T = k, Z]`, combined into the cross-fitted AIPW score
//!   `phi_k = m_k(Z) + 1{T = k} (Y - m_k(Z)) / e_k(Z)` whose mean estimates `E[Y(k)]` for the
//!   whole population. Any two levels then contrast over that same population, so the
//!   contrasts are comparable with each other.
//!
//! The two targets differ whenever the effect varies with the covariates, because they
//! average it over different populations. Neither is chosen by default.
//!
//! The per-level scores are one [`ScoreTable`] column each, so the means, their joint
//! covariance and any contrast's standard error come from the table's influence-function
//! summary. As for binary AIPW, the standard error is the plug-in iid score variance of
//! cross-fitted scores: it is meaningful under consistency, positivity (no level has a
//! generalized propensity near zero) and a small product of nuisance errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, OutcomeFunctional, TargetPopulation,
};
use antecedent_data::TabularData;
use antecedent_expr::IdentifiedEstimand;
use antecedent_stats::{
    DenseLinearAlgebra, FaerBackend, GlmOptions, LeastSquaresWorkspace, MultinomialDesignRef,
    fit_multinomial_logit_weighted,
};

use super::levels::{OtherLevels, describe_observed_levels};
use super::prepare::{clip_of, default_propensity_overlap};
use crate::adjustment::EffectEstimate;
use crate::aipw::{AipwAte, AipwWorkspace, predict_colmajor};
use crate::crossfit_aipw::DEFAULT_AIPW_FOLDS;
use crate::crossfit_plan::{
    CrossfitPlan, FoldRows, FoldSource, SHARED_FOLD_ID_ERRORS, rows_colmajor,
};
use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;
use crate::scores::{ScoreColumn, ScoreTable};
use crate::util::refuse;

/// Provenance tag frozen on the generalized-propensity score table.
pub const GENERALIZED_PROPENSITY_PROVENANCE: &str = "gps.crossfit.multinomial_logit.ols.v1";

/// Which population a multi-level observational contrast is estimated over.
#[derive(Clone, Debug, PartialEq)]
pub enum MultiLevelTarget {
    /// The query's active and control levels on the rows at one of the two (`T in {a, b}`):
    /// the pairwise-subpopulation effect. Rows at other levels are excluded and counted.
    PairwiseSubpopulation,
    /// The whole complete-case population, standardized with a multinomial-logit
    /// generalized propensity over `levels`. Every row must be at one of the declared levels.
    FullPopulation {
        /// The declared treatment level set, at least two distinct finite values containing
        /// the query's active and control levels.
        levels: Arc<[f64]>,
    },
}

impl MultiLevelTarget {
    /// Stable spelling of the semantics.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::PairwiseSubpopulation => "pairwise_subpopulation",
            Self::FullPopulation { .. } => "full_population_generalized_propensity",
        }
    }

    /// The population the contrast speaks for.
    #[must_use]
    pub const fn population(&self) -> MultiLevelPopulation {
        match self {
            Self::PairwiseSubpopulation => MultiLevelPopulation::RowsAtTheTwoDeclaredLevels,
            Self::FullPopulation { .. } => MultiLevelPopulation::AllCompleteCaseRows,
        }
    }
}

/// The population a multi-level contrast is an average over.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MultiLevelPopulation {
    /// Rows whose treatment is at the active or the control level.
    RowsAtTheTwoDeclaredLevels,
    /// Every complete-case row, whatever its observed treatment level.
    AllCompleteCaseRows,
}

impl MultiLevelPopulation {
    /// Stable spelling of the population.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RowsAtTheTwoDeclaredLevels => "rows at the two declared levels",
            Self::AllCompleteCaseRows => "all complete-case rows",
        }
    }
}

/// Rows observed at one treatment level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevelRows {
    /// Treatment level.
    pub level: f64,
    /// Complete-case rows observed at it.
    pub rows: u64,
}

/// The standardized mean outcome `E[Y(level)]` of one level over the full population.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevelMean {
    /// Treatment level.
    pub level: f64,
    /// Cross-fitted AIPW mean.
    pub mean: f64,
    /// Influence-function standard error of the mean.
    pub se: f64,
    /// Smallest and largest out-of-fold generalized propensity of the level over all rows,
    /// before any clip.
    pub propensity_range: (f64, f64),
}

/// A multi-level observational contrast with the population it speaks for.
#[derive(Clone, Debug)]
pub struct MultiLevelEstimate {
    /// The active-minus-control contrast, its standard error, score table (full-population
    /// target) and diagnostics.
    pub effect: EffectEstimate,
    /// The semantics the estimate was computed under.
    pub target: MultiLevelTarget,
    /// The population the contrast is an average over, stated for the reader.
    pub target_population: MultiLevelPopulation,
    /// Complete-case rows at each declared level (the pair, or the full level set).
    pub rows_by_level: Vec<LevelRows>,
    /// Complete-case rows at no declared level that the target excluded; always zero for the
    /// full-population target, which refuses them.
    pub excluded_rows: u64,
    /// Standardized mean outcome per declared level; empty for the pairwise target.
    pub level_means: Vec<LevelMean>,
}

/// Cross-fitted AIPW for contrasts between levels of a multi-level treatment.
#[derive(Clone, Debug)]
pub struct MultiLevelAipw {
    /// The population semantics. There is no default: it changes the estimand.
    pub target: MultiLevelTarget,
    /// Cross-fit folds (full-population target).
    pub folds: usize,
    /// GLM options for the propensity model.
    pub glm_options: GlmOptions,
    /// Overlap policy; a clip floors the generalized propensities, and a trim is refused.
    pub overlap: OverlapPolicy,
}

impl MultiLevelAipw {
    /// The pairwise-subpopulation effect of the query's two levels.
    #[must_use]
    pub fn pairwise_subpopulation() -> Self {
        Self::with_target(MultiLevelTarget::PairwiseSubpopulation)
    }

    /// The full-population effect, standardized over the declared level set.
    #[must_use]
    pub fn full_population(levels: impl Into<Arc<[f64]>>) -> Self {
        Self::with_target(MultiLevelTarget::FullPopulation { levels: levels.into() })
    }

    fn with_target(target: MultiLevelTarget) -> Self {
        Self {
            target,
            folds: DEFAULT_AIPW_FOLDS,
            glm_options: GlmOptions::default(),
            overlap: default_propensity_overlap(),
        }
    }

    /// Set the overlap policy.
    #[must_use]
    pub const fn with_overlap(mut self, overlap: OverlapPolicy) -> Self {
        self.overlap = overlap;
        self
    }

    /// Set the cross-fit fold count of the full-population target.
    #[must_use]
    pub const fn with_folds(mut self, folds: usize) -> Self {
        self.folds = folds;
        self
    }

    /// Estimate the active-minus-control contrast of `query` under the declared target.
    ///
    /// # Errors
    ///
    /// A query outside the route (effect modifiers, a non-mean functional, a target
    /// population other than `AllObserved`), an explicit overlap override or trim, rows at an
    /// undeclared level (full-population target), a declared level with no rows, a
    /// cross-fit training fold missing a level or with too few rows for its outcome
    /// regression, a generalized propensity of zero with no clip, or a GLM/OLS failure.
    pub fn fit(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<MultiLevelEstimate, EstimationError> {
        if !estimand.is_adjustment_shaped() {
            return Err(EstimationError::IncompatibleEstimand {
                message: "multi-level treatment estimators expect an adjustment-shaped estimand",
            });
        }
        query.validate()?;
        if !query.effect_modifiers.is_empty() {
            return Err(EstimationError::EffectModifiers);
        }
        if query.outcome_functional != OutcomeFunctional::Mean {
            return Err(EstimationError::unsupported_coded(
                antecedent_core::reason_code!("route_not_supported"),
                "multi-level treatment estimators support the mean functional only",
            ));
        }
        if query.target_population != TargetPopulation::AllObserved {
            return Err(EstimationError::TargetPopulation);
        }
        match &self.target {
            MultiLevelTarget::PairwiseSubpopulation => {
                self.fit_pairwise(data, estimand, query, ctx, assumptions)
            }
            MultiLevelTarget::FullPopulation { levels } => {
                self.fit_full_population(levels, data, estimand, query, ctx, assumptions)
            }
        }
    }

    fn fit_pairwise(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<MultiLevelEstimate, EstimationError> {
        let aipw = AipwAte::new()
            .with_bootstrap_replicates(0)
            .with_overlap(self.overlap)
            .with_glm_options(self.glm_options)
            .with_other_levels(OtherLevels::ExcludeUndeclared);
        let problem = aipw.prepare(data, estimand, query)?;
        let effect = aipw.fit(&problem, &mut AipwWorkspace::default(), ctx, assumptions)?;
        let levels = problem.treatment_levels;
        Ok(MultiLevelEstimate {
            effect,
            target: self.target.clone(),
            target_population: self.target.population(),
            rows_by_level: vec![
                LevelRows { level: levels.active, rows: levels.rows_at_active },
                LevelRows { level: levels.control, rows: levels.rows_at_control },
            ],
            excluded_rows: levels.excluded_rows,
            level_means: Vec::new(),
        })
    }

    fn fit_full_population(
        &self,
        levels: &[f64],
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<MultiLevelEstimate, EstimationError> {
        crate::util::refuse_explicit_override(
            self.overlap,
            "generalized-propensity estimators require RequireDiagnostics overlap policy; positivity is mandatory",
        )?;
        self.overlap.validate()?;
        if crate::propensity::trim_of(self.overlap).is_some() {
            return Err(EstimationError::unsupported_coded(
                antecedent_core::reason_code!("option_not_applicable"),
                "generalized-propensity estimators do not trim: trimming one level's propensity would redefine the common-support population of every contrast",
            ));
        }
        let (active, control, _) =
            crate::prepare::treatment_contrast(&query.active, &query.control)?;
        let prepared = PreparedLevels::new(data, estimand, query, levels)?;
        let active_index = prepared.index_of(active)?;
        let control_index = prepared.index_of(control)?;
        let table = self.crossfit_scores(&prepared, query, ctx)?;

        let summary = table.summarize(None)?;
        let mut coefficients = vec![0.0; levels.len()];
        coefficients[active_index] = 1.0;
        coefficients[control_index] = -1.0;
        let contrast = table.linear_contrast(&summary, &coefficients)?;
        let influence = table.combine_scores(&coefficients)?;

        let mut level_means = Vec::with_capacity(levels.len());
        for (index, &level) in levels.iter().enumerate() {
            let propensities =
                &table.propensities[index * table.n_rows..(index + 1) * table.n_rows];
            level_means.push(LevelMean {
                level,
                mean: summary.means[index],
                se: summary.covariance.se(index),
                propensity_range: propensities
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &e| {
                        (lo.min(e), hi.max(e))
                    }),
            });
        }
        let effect = EffectEstimate::new(contrast.value, contrast.se, assumptions, self.overlap)
            .with_n_obs(u64::try_from(table.n_rows).unwrap_or(u64::MAX))
            .with_preparation(Some(crate::prepare::preparation_summary(&prepared.preparation)))
            .with_influence(Some(Arc::from(influence)))
            .with_score_table(Some(table));
        Ok(MultiLevelEstimate {
            effect,
            target: self.target.clone(),
            target_population: self.target.population(),
            rows_by_level: levels
                .iter()
                .zip(&prepared.rows_by_level)
                .map(|(&level, &rows)| LevelRows { level, rows })
                .collect(),
            excluded_rows: 0,
            level_means,
        })
    }

    /// Cross-fitted per-level AIPW scores: one column per declared level, in declaration order.
    #[allow(
        clippy::too_many_lines,
        reason = "one cross-fitting loop: fit the multinomial propensity on the training rows, then score every level on the validation rows with its own outcome regression"
    )]
    fn crossfit_scores(
        &self,
        prepared: &PreparedLevels,
        query: &AverageEffectQuery,
        ctx: &ExecutionContext,
    ) -> Result<ScoreTable, EstimationError> {
        let n = prepared.n;
        let ncols = prepared.ncols;
        let k = prepared.levels.len();
        if self.folds < 2 || n < self.folds {
            return Err(EstimationError::data_msg(
                "generalized-propensity cross-fitting requires folds in 2..=n",
            ));
        }
        let plan = CrossfitPlan::derive(
            n,
            self.folds,
            FoldSource::Stratified {
                strata: &prepared.category,
                units: &prepared.row_index,
                seed: ctx.rng.master_seed(),
            },
            &SHARED_FOLD_ID_ERRORS,
        )?;
        let clip = clip_of(self.overlap);
        let floor = clip.unwrap_or(0.0);
        let mut scores = vec![0.0; n * k];
        let mut propensities = vec![0.0; n * k];
        let mut glm_workspace = LeastSquaresWorkspace::default();
        let mut ols_workspace = LeastSquaresWorkspace::default();
        let mut prediction = Vec::new();

        plan.run(
            None,
            |_fold| {
                if ctx.cancellation.is_cancelled() {
                    return Err(refuse(
                        antecedent_core::reason_code!("cancelled_no_claim"),
                        "treatment_levels.cancelled",
                        "the cross-fitted generalized-propensity fit was cancelled before every fold was fit; no estimate is reported",
                    ));
                }
                Ok(())
            },
            |FoldRows { train, valid, .. }| {
                let train_category: Vec<u32> =
                    train.iter().map(|&i| prepared.category[i]).collect();
                let mut present = vec![false; k];
                for &c in &train_category {
                    present[c as usize] = true;
                }
                if let Some(missing) = present.iter().position(|&seen| !seen) {
                    return Err(refuse(
                        antecedent_core::reason_code!("arm_not_populated"),
                        "treatment_levels.training_fold_missing_level",
                        &format!(
                            "a cross-fit training fold has no row at treatment level {}; use fewer folds or merge the level",
                            prepared.levels[missing]
                        ),
                    ));
                }
                let design_valid = rows_colmajor(&prepared.design, n, ncols, valid);
                let design_train = rows_colmajor(&prepared.design, n, ncols, train);
                let e_valid = fit_category_probabilities(
                    &CategoryTrainingRows {
                        design: &design_train,
                        category: &train_category,
                        n_categories: k,
                        ncols,
                    },
                    &design_valid,
                    valid.len(),
                    &self.glm_options,
                    &mut glm_workspace,
                )?;
                // The clip is the only floor. With none, a generalized propensity of exactly 0
                // (infinite weight) is refused rather than floored at a hidden constant.
                if clip.is_none() && e_valid.iter().any(|&e| e.is_nan() || e <= 0.0) {
                    return Err(EstimationError::OverlapWithFields {
                        message: "a fitted generalized propensity is 0 (or not finite) and no clip is set; the inverse-probability weight is infinite — set an overlap clip",
                        fields: Box::new(crate::overlap::propensity_score_fields(&e_valid)),
                    });
                }
                Ok((design_valid, e_valid))
            },
            |FoldRows { train, valid, .. }, (design_valid, e_valid)| {
                let nv = valid.len();
                for c in 0..k {
                    let level_train: Vec<usize> = train
                        .iter()
                        .copied()
                        .filter(|&i| prepared.category[i] as usize == c)
                        .collect();
                    // The per-level OLS needs residual degrees of freedom; an exact interpolant
                    // would leave double robustness resting on the propensity alone.
                    if level_train.len() <= ncols {
                        return Err(refuse(
                            antecedent_core::reason_code!("arm_not_populated"),
                            "treatment_levels.sparse_level",
                            &format!(
                                "a cross-fit training fold has no more rows at treatment level {} than design columns, so its outcome model has no residual degrees of freedom",
                                prepared.levels[c]
                            ),
                        ));
                    }
                    let design_level = rows_colmajor(&prepared.design, n, ncols, &level_train);
                    let y_level: Vec<f64> =
                        level_train.iter().map(|&i| prepared.outcome[i]).collect();
                    let outcome_fit = FaerBackend
                        .least_squares(
                            &design_level,
                            level_train.len(),
                            ncols,
                            &y_level,
                            &mut ols_workspace,
                        )
                        .map_err(crate::error::EstimationError::from)?;
                    predict_colmajor(
                        &design_valid,
                        nv,
                        ncols,
                        &outcome_fit.coefficients,
                        &mut prediction,
                    );
                    for (r, &i) in valid.iter().enumerate() {
                        let e = e_valid[c * nv + r];
                        propensities[c * n + i] = e;
                        let at_level = f64::from(prepared.category[i] as usize == c);
                        scores[c * n + i] = prediction[r]
                            + at_level / e.max(floor) * (prepared.outcome[i] - prediction[r]);
                    }
                }
                Ok(())
            },
        )?;

        let columns: Vec<ScoreColumn> = (0..k)
            .map(|c| ScoreColumn { arm: u32::try_from(c).unwrap_or(u32::MAX), threshold: None })
            .collect();
        Ok(ScoreTable {
            observed_arm: Arc::from(prepared.category.as_slice()),
            propensities: propensities.into(),
            observed_outcome: Arc::from(prepared.outcome.as_slice()),
            n_rows: n,
            row_index: Arc::from(prepared.row_index.as_slice()),
            fold_ids: Arc::from(plan.into_fold_ids()),
            n_folds: u32::try_from(self.folds).unwrap_or(u32::MAX),
            scores: Arc::from(scores),
            columns: Arc::from(columns),
            adjustment_set: Arc::clone(&prepared.adjustment_set),
            nuisance_provenance: Arc::from(GENERALIZED_PROPENSITY_PROVENANCE),
            propensity_clip: clip,
            treatment: query.treatment,
            intervened: Arc::from([]),
            preparation: None,
        })
    }
}

/// Training rows of a categorical (multinomial-logit) fit.
pub(crate) struct CategoryTrainingRows<'a> {
    /// Column-major `[1 | Z...]` design of the training rows.
    pub design: &'a [f64],
    /// Category index of each training row.
    pub category: &'a [u32],
    /// Number of categories.
    pub n_categories: usize,
    /// Design columns.
    pub ncols: usize,
}

/// Fit a multinomial logit on `train` and return the category probabilities of `valid_rows`
/// design rows, category `c`, row `r` at `c * valid_rows + r`.
///
/// # Errors
///
/// A GLM failure: separation, non-convergence or a singular design.
pub(crate) fn fit_category_probabilities(
    train: &CategoryTrainingRows<'_>,
    design_valid: &[f64],
    valid_rows: usize,
    options: &GlmOptions,
    workspace: &mut LeastSquaresWorkspace,
) -> Result<Vec<f64>, EstimationError> {
    let fit = fit_multinomial_logit_weighted(
        MultinomialDesignRef {
            x_colmajor: train.design,
            nrows: train.category.len(),
            ncols: train.ncols,
            y_category: train.category,
            n_categories: train.n_categories,
        },
        None,
        &FaerBackend,
        workspace,
        options,
    )
    .map_err(crate::error::EstimationError::from)?;
    fit.require_ok().map_err(crate::error::EstimationError::from)?;
    Ok(multinomial_probabilities(
        &fit.coefficients,
        design_valid,
        valid_rows,
        train.ncols,
        train.n_categories,
    ))
}

/// Softmax probabilities of a baseline-category multinomial logit on the rows of a
/// column-major design: category `c`, row `r` at `c * rows + r`.
///
/// `coefficients` is row-major `[k * ncols]` with category 0 all zeros (the layout of
/// `MultinomialFit::coefficients`).
fn multinomial_probabilities(
    coefficients: &[f64],
    design: &[f64],
    rows: usize,
    ncols: usize,
    k: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; rows * k];
    let mut eta = vec![0.0; k];
    for r in 0..rows {
        for (c, slot) in eta.iter_mut().enumerate() {
            *slot = (0..ncols).map(|j| design[j * rows + r] * coefficients[c * ncols + j]).sum();
        }
        let top = eta.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mass: f64 = eta.iter().map(|&e| (e - top).exp()).sum();
        for (c, &e) in eta.iter().enumerate() {
            out[c * rows + r] = (e - top).exp() / mass;
        }
    }
    out
}

/// The complete-case design of a full-population fit: treatment coded as an index into the
/// declared level set.
struct PreparedLevels {
    n: usize,
    ncols: usize,
    /// Column-major `[1 | Z...]`.
    design: Vec<f64>,
    levels: Vec<f64>,
    /// Index into `levels` of each row's treatment.
    category: Vec<u32>,
    outcome: Vec<f64>,
    row_index: Vec<antecedent_data::GlobalId>,
    adjustment_set: Arc<[antecedent_core::VariableId]>,
    rows_by_level: Vec<u64>,
    preparation: antecedent_data::materialize::PreparationReceipt,
}

impl PreparedLevels {
    #[allow(
        clippy::float_cmp,
        reason = "declared treatment levels are exact category labels, not computed values"
    )]
    fn new(
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
        levels: &[f64],
    ) -> Result<Self, EstimationError> {
        let mut sorted = levels.to_vec();
        sorted.sort_by(f64::total_cmp);
        if levels.len() < 2
            || levels.iter().any(|l| !l.is_finite())
            || sorted.windows(2).any(|pair| pair[0] == pair[1])
        {
            return Err(refuse(
                antecedent_core::reason_code!("invalid_argument"),
                "treatment_levels.invalid_level_set",
                "the declared treatment levels must be at least two distinct finite values",
            ));
        }
        let mut ids = Vec::with_capacity(2 + estimand.adjustment_set.len());
        ids.push(query.treatment);
        ids.push(query.outcome);
        ids.extend_from_slice(&estimand.adjustment_set);
        let crate::prepare::AdmittedColumns { data, row_mask: mask, receipt: preparation } =
            crate::prepare::admit_estimator_columns(data, &ids, crate::prepare::FINITE_TABLE_ONLY)?;
        let t = data.float64_selected(query.treatment, &mask).map_err(EstimationError::from)?;
        let outcome = data
            .float64_selected(query.outcome, &mask)
            .map_err(EstimationError::from)?
            .into_owned();
        let n = t.len();
        if n == 0 {
            return Err(EstimationError::data_msg(
                "no complete-case rows for the generalized-propensity design",
            ));
        }
        let mut category = Vec::with_capacity(n);
        let mut rows_by_level = vec![0u64; levels.len()];
        let mut undeclared = 0u64;
        let mut first_undeclared = None;
        for &value in t.iter() {
            if let Some(index) = levels.iter().position(|&l| l == value) {
                category.push(u32::try_from(index).unwrap_or(u32::MAX));
                rows_by_level[index] += 1;
            } else {
                undeclared += 1;
                first_undeclared.get_or_insert(value);
            }
        }
        if let Some(value) = first_undeclared {
            return Err(refuse(
                antecedent_core::reason_code!("treatment_level_undeclared"),
                "treatment_levels.full_population_undeclared_level",
                &format!(
                    "{undeclared} complete-case rows have a treatment value outside the declared \
                     level set; first found {value}; {}. The full-population estimate averages \
                     over every row, so each row must be at a declared level: declare the \
                     missing level, or use the pairwise-subpopulation target to restrict the \
                     population to the rows at two levels",
                    describe_observed_levels(&t)
                ),
            ));
        }
        if let Some(empty) = rows_by_level.iter().position(|&rows| rows == 0) {
            return Err(refuse(
                antecedent_core::reason_code!("arm_not_populated"),
                "treatment_levels.level_not_populated",
                &format!("no complete-case row has the declared treatment level {}", levels[empty]),
            ));
        }
        let ncols = 1 + estimand.adjustment_set.len();
        let mut design = vec![0.0; n * ncols];
        design[..n].fill(1.0);
        for (j, &z) in estimand.adjustment_set.iter().enumerate() {
            data.float64_selected_into(z, &mask, &mut design[(1 + j) * n..(2 + j) * n])
                .map_err(EstimationError::from)?;
        }
        let mut row_index = Vec::with_capacity(n);
        for (i, _) in mask.iter().enumerate().filter(|&(_, &keep)| keep) {
            row_index.push(antecedent_data::GlobalId::new(u64::try_from(i).map_err(|_| {
                        EstimationError::data_msg(
                            "address the row by GlobalId and keep ChunkIndex inside one bounded chunk. A u32 backend feature flag is not that attempt",
                        )
                    })?));
        }
        Ok(Self {
            n,
            ncols,
            design,
            levels: levels.to_vec(),
            category,
            outcome,
            row_index,
            adjustment_set: Arc::clone(&estimand.adjustment_set),
            rows_by_level,
            preparation,
        })
    }

    /// Index in the declared level set of a contrast level.
    #[allow(
        clippy::float_cmp,
        reason = "declared treatment levels are exact category labels, not computed values"
    )]
    fn index_of(&self, level: f64) -> Result<usize, EstimationError> {
        self.levels.iter().position(|&l| l == level).ok_or_else(|| {
            refuse(
                antecedent_core::reason_code!("invalid_argument"),
                "treatment_levels.contrast_level_not_declared",
                &format!("the query contrasts treatment level {level}, which is not in the declared level set"),
            )
        })
    }
}
