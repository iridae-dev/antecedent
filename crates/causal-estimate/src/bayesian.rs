//! Bayesian mechanisms, g-computation, and posterior functional evaluation (Phase 6).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::needless_pass_by_value,
    clippy::doc_markdown,
    clippy::many_single_char_names
)]

use std::sync::Arc;

use causal_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, AverageEffectQuery, ExecutionContext, TargetPopulation, VariableId,
};
use causal_data::TabularData;
use causal_expr::IdentifiedEstimand;
use causal_identify::IdentificationStatus;
use causal_prob::{
    BayesDesignRef, BayesFitOptions, BayesLikelihood, ConjugateGaussianBackend, EffectBatch,
    InferenceBackend, InferenceDiagnostics, LaplaceGlmBackend, LaplaceWorkspace, PosteriorBatch,
    PosteriorDraws, PosteriorEvalWorkspace, PosteriorQuantityKind, PosteriorSchema,
    PosteriorSummary, PriorSensitivitySummary, PriorSet, PriorSpec,
};
use causal_stats::{CompiledDesign, GlmFamily};

use crate::adjustment::{OverlapPolicy, intervention_f64};
use crate::error::EstimationError;
use crate::util::require_explicit_override;

/// Causal posterior over an identified functional (DESIGN.md §14.4).
#[derive(Clone, Debug)]
pub struct CausalPosterior {
    /// Columnar effect (and optional coefficient) draws.
    pub draws: PosteriorDraws,
    /// Summary of `draws`.
    pub summaries: PosteriorSummary,
    /// Identification status — priors never upgrade this.
    pub identification: IdentificationStatus,
    /// Optional prior-sensitivity grid.
    pub prior_sensitivity: Option<PriorSensitivitySummary>,
    /// Inference diagnostics.
    pub diagnostics: InferenceDiagnostics,
    /// Assumptions including prior restrictions.
    pub assumptions: AssumptionSet,
    /// Unidentified graph mass retained when aggregating envelopes (0 if single graph).
    pub unidentified_mass: f64,
}

impl CausalPosterior {
    /// Primary effect column index (first `Effect` quantity), if any.
    #[must_use]
    pub fn effect_column(&self) -> Option<usize> {
        self.draws
            .schema
            .quantities
            .iter()
            .position(|q| matches!(q, PosteriorQuantityKind::Effect { .. }))
    }

    /// Empirical P(effect < threshold) for the primary effect column.
    ///
    /// # Errors
    ///
    /// Missing effect column.
    pub fn probability_below(&self, threshold: f64) -> Result<f64, EstimationError> {
        let q = self
            .effect_column()
            .ok_or_else(|| EstimationError::Stats("CausalPosterior has no effect column".into()))?;
        self.draws
            .probability_below(q, threshold)
            .map_err(|e| EstimationError::Stats(e.to_string()))
    }
}

/// Bayesian linear / GLM mechanism fit (coefficient posterior).
#[derive(Clone, Debug)]
pub struct BayesianGlmMechanism {
    /// Fitted coefficient draws (columnar).
    pub coefficient_draws: PosteriorDraws,
    /// MAP / posterior mode coefficients.
    pub map: Vec<f64>,
    /// Likelihood used.
    pub likelihood: BayesLikelihood,
    /// Diagnostics.
    pub diagnostics: InferenceDiagnostics,
    /// Compiled design retained for g-computation.
    pub design: CompiledDesign,
    /// Treatment column index in the design.
    pub treatment_col: usize,
    /// Active / control levels.
    pub active: f64,
    /// Control level.
    pub control: f64,
}

/// Which inference backend to use for Bayesian g-computation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BayesianBackendKind {
    /// Analytic conjugate Gaussian (identity link only).
    ConjugateGaussian,
    /// Native Laplace GLM.
    Laplace,
}

/// Bayesian g-computation ATE estimator.
#[derive(Clone, Debug)]
pub struct BayesianGComputationAte {
    /// Backend kind.
    pub backend: BayesianBackendKind,
    /// Likelihood (Laplace); conjugate forces GaussianIdentity.
    pub likelihood: BayesLikelihood,
    /// Draw count.
    pub n_draws: usize,
    /// RNG seed.
    pub seed: u64,
    /// Overlap policy (must be ExplicitOverride).
    pub overlap: OverlapPolicy,
    /// Prior scale for isotropic Gaussian coefficients (weakly informative default 10).
    pub prior_scale: f64,
}

impl Default for BayesianGComputationAte {
    fn default() -> Self {
        Self::new()
    }
}

impl BayesianGComputationAte {
    /// Laplace Gaussian defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: BayesianBackendKind::Laplace,
            likelihood: BayesLikelihood::GaussianIdentity,
            n_draws: 1000,
            seed: 0,
            overlap: OverlapPolicy::ExplicitOverride,
            prior_scale: 10.0,
        }
    }

    /// Conjugate Gaussian linear path.
    #[must_use]
    pub fn conjugate() -> Self {
        Self {
            backend: BayesianBackendKind::ConjugateGaussian,
            likelihood: BayesLikelihood::GaussianIdentity,
            ..Self::new()
        }
    }

    /// Prepare from data + identified estimand (same IR as frequentist adjustment).
    ///
    /// # Errors
    ///
    /// Overlap / estimand / data failures.
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedBayesianProblem, EstimationError> {
        require_explicit_override(
            self.overlap,
            "BayesianGComputationAte requires ExplicitOverride overlap policy",
        )?;
        if &*estimand.method != "backdoor.adjustment" && &*estimand.method != "backdoor.efficient" {
            return Err(EstimationError::IncompatibleEstimand {
                message: "BayesianGComputationAte expects backdoor.adjustment/efficient",
            });
        }
        query.validate().map_err(|e| EstimationError::UnsupportedQuery(e.to_string()))?;
        if !query.effect_modifiers.is_empty() {
            return Err(EstimationError::UnsupportedQuery(
                "Bayesian g-comp does not support effect modifiers".into(),
            ));
        }
        if query.target_population != TargetPopulation::AllObserved {
            return Err(EstimationError::UnsupportedQuery(
                "Bayesian g-comp only supports TargetPopulation::AllObserved".into(),
            ));
        }
        let active = intervention_f64(&query.active)?;
        let control = intervention_f64(&query.control)?;
        if (active - control).abs() < f64::EPSILON {
            return Err(EstimationError::UnsupportedQuery(
                "active and control treatment levels must differ".into(),
            ));
        }

        let treatment = query.treatment;
        let outcome = query.outcome;
        let mut ids = Vec::with_capacity(2 + estimand.adjustment_set.len());
        ids.push(treatment);
        ids.push(outcome);
        ids.extend_from_slice(&estimand.adjustment_set);
        let row_mask =
            data.complete_case_mask(&ids).map_err(|e| EstimationError::Data(e.to_string()))?;
        let t = data
            .float64_masked(treatment, &row_mask)
            .map_err(|e| EstimationError::Data(e.to_string()))?;
        let y = data
            .float64_masked(outcome, &row_mask)
            .map_err(|e| EstimationError::Data(e.to_string()))?;
        let mut covs: Vec<(VariableId, Vec<f64>)> = Vec::new();
        for &z in estimand.adjustment_set.iter() {
            covs.push((
                z,
                data.float64_masked(z, &row_mask)
                    .map_err(|e| EstimationError::Data(e.to_string()))?,
            ));
        }
        let cov_refs: Vec<(VariableId, &[f64])> =
            covs.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        let selected_rows: Vec<usize> =
            row_mask.iter().enumerate().filter_map(|(i, keep)| keep.then_some(i)).collect();
        let design = CompiledDesign::linear_adjustment(&t, &cov_refs, &y, &selected_rows)
            .map_err(|e| EstimationError::Stats(e.to_string()))?;
        Ok(PreparedBayesianProblem {
            design,
            method: Arc::clone(&estimand.method),
            adjustment_set: Arc::clone(&estimand.adjustment_set),
            active,
            control,
            overlap: self.overlap,
        })
    }

    /// Fit mechanism + evaluate ATE g-computation posterior.
    ///
    /// `identification` is recorded as-is; informative priors never change it.
    ///
    /// # Errors
    ///
    /// Backend / evaluation failures.
    pub fn fit(
        &self,
        problem: &PreparedBayesianProblem,
        identification: IdentificationStatus,
        workspace: &mut BayesianGCompWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CausalPosterior, EstimationError> {
        let prior = PriorSet {
            specs: vec![PriorSpec::GaussianCoefficients(
                causal_prob::GaussianCoefficientPrior::isotropic(
                    problem.design.ncols,
                    self.prior_scale,
                ),
            )],
            contrast: None,
            categorical: Vec::new(),
        };
        let mut assumptions = AssumptionSet::new();
        for spec in &prior.specs {
            assumptions.push(AssumptionRecord {
                assumption: Assumption::PriorRestriction(spec.as_assumption()),
                source: AssumptionSource::AlgorithmDefault {
                    algorithm: Arc::from("bayesian_gcomp"),
                },
                scope: AssumptionScope::Estimation,
                status: AssumptionStatus::Untestable,
            });
        }

        let likelihood = match self.backend {
            BayesianBackendKind::ConjugateGaussian => BayesLikelihood::GaussianIdentity,
            BayesianBackendKind::Laplace => self.likelihood,
        };
        let opts = BayesFitOptions {
            n_draws: self.n_draws,
            seed: self.seed,
            ..BayesFitOptions::default()
        };
        let design_ref = BayesDesignRef {
            x_colmajor: &problem.design.matrix,
            nrows: problem.design.nrows,
            ncols: problem.design.ncols,
            y: &problem.design.outcome,
            weights: None,
            offsets: None,
        };

        let fit = match self.backend {
            BayesianBackendKind::ConjugateGaussian => ConjugateGaussianBackend.fit(
                likelihood,
                design_ref,
                &prior,
                &opts,
                &mut workspace.laplace,
                ctx,
            ),
            BayesianBackendKind::Laplace => LaplaceGlmBackend.fit(
                likelihood,
                design_ref,
                &prior,
                &opts,
                &mut workspace.laplace,
                ctx,
            ),
        }
        .map_err(prob_err)?;

        if !fit.diagnostics.allows_posterior() {
            return Err(EstimationError::Stats("Bayesian fit refused without diagnostics".into()));
        }

        let t_col = problem
            .design
            .treatment_column()
            .ok_or_else(|| EstimationError::Stats("missing treatment column".into()))?;

        let mechanism = BayesianGlmMechanism {
            coefficient_draws: fit.draws,
            map: fit.map,
            likelihood,
            diagnostics: fit.diagnostics.clone(),
            design: problem.design.clone(),
            treatment_col: t_col,
            active: problem.active,
            control: problem.control,
        };

        let glm_family = likelihood_to_glm_family(likelihood);
        let evaluator = GCompAteEvaluator {
            family: glm_family,
            treatment_col: t_col,
            active: problem.active,
            control: problem.control,
            nrows: problem.design.nrows,
            ncols: problem.design.ncols,
            matrix: Arc::clone(&problem.design.matrix),
        };
        let compiled = evaluator.compile()?;
        let n_draws = mechanism.coefficient_draws.n_draws;
        workspace.eval.prepare(n_draws, problem.design.ncols);
        let mut effect_out = EffectBatch::default();
        effect_out.prepare(n_draws);
        let batch = mechanism
            .coefficient_draws
            .batch(0, n_draws)
            .map_err(|e| EstimationError::Stats(e.to_string()))?;
        evaluator.evaluate_batch(&compiled, batch, &mut effect_out, &mut workspace.eval, ctx)?;

        let mut quantities = mechanism.coefficient_draws.schema.quantities.to_vec();
        // Drop residual variance column from combined effect artifact if present — keep coefs + effect.
        quantities.retain(|q| !matches!(q, PosteriorQuantityKind::ResidualVariance));
        let effect_idx = quantities.len();
        quantities.push(PosteriorQuantityKind::Effect { name: Arc::from("ate") });
        let n_q = quantities.len();
        let mut values = vec![0.0; n_draws * n_q];
        for (qi, q) in mechanism.coefficient_draws.schema.quantities.iter().enumerate() {
            if matches!(q, PosteriorQuantityKind::ResidualVariance) {
                continue;
            }
            let dest = quantities.iter().position(|qq| qq == q).expect("quantity present");
            let col = mechanism
                .coefficient_draws
                .column(qi)
                .map_err(|e| EstimationError::Stats(e.to_string()))?;
            values[dest * n_draws..(dest + 1) * n_draws].copy_from_slice(col);
        }
        values[effect_idx * n_draws..(effect_idx + 1) * n_draws]
            .copy_from_slice(&effect_out.values[..n_draws]);

        let draws = PosteriorDraws::from_column_major(
            PosteriorSchema { quantities: Arc::from(quantities) },
            n_draws,
            values,
        )
        .map_err(|e| EstimationError::Stats(e.to_string()))?;
        let summaries = draws.summarize();

        let _ = mechanism;
        Ok(CausalPosterior {
            draws,
            summaries,
            identification,
            prior_sensitivity: None,
            diagnostics: fit.diagnostics,
            assumptions,
            unidentified_mass: 0.0,
        })
    }
}

/// Prepared Bayesian g-comp problem.
#[derive(Clone, Debug)]
pub struct PreparedBayesianProblem {
    /// Design.
    pub design: CompiledDesign,
    /// Estimand method.
    pub method: Arc<str>,
    /// Adjustment set.
    pub adjustment_set: Arc<[VariableId]>,
    /// Active treatment.
    pub active: f64,
    /// Control treatment.
    pub control: f64,
    /// Overlap.
    pub overlap: OverlapPolicy,
}

/// Workspace for Bayesian g-comp.
#[derive(Clone, Debug, Default)]
pub struct BayesianGCompWorkspace {
    /// Laplace / conjugate workspace.
    pub laplace: LaplaceWorkspace,
    /// Posterior functional eval scratch.
    pub eval: PosteriorEvalWorkspace,
}

/// Trait for batched posterior functional evaluation (DESIGN.md §14.4).
pub trait PosteriorFunctionalEvaluator {
    /// Compiled plan type.
    type Compiled;

    /// Compile against a posterior schema.
    ///
    /// # Errors
    ///
    /// Incompatible schema.
    fn compile(&self) -> Result<Self::Compiled, EstimationError>;

    /// Evaluate a batch of coefficient draws into effects.
    ///
    /// # Errors
    ///
    /// Shape / numerical failures.
    fn evaluate_batch(
        &self,
        compiled: &Self::Compiled,
        posterior: PosteriorBatch<'_>,
        output: &mut EffectBatch,
        workspace: &mut PosteriorEvalWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<(), EstimationError>;
}

/// Compiled g-comp ATE evaluator (finite-difference mean contrast).
#[derive(Clone, Debug)]
pub struct GCompAteEvaluator {
    /// Mean family.
    pub family: GlmFamily,
    /// Treatment column.
    pub treatment_col: usize,
    /// Active level.
    pub active: f64,
    /// Control level.
    pub control: f64,
    /// Rows.
    pub nrows: usize,
    /// Cols.
    pub ncols: usize,
    /// Design matrix (column-major).
    pub matrix: Arc<[f64]>,
}

/// Empty compiled marker (evaluator is self-contained).
#[derive(Clone, Copy, Debug, Default)]
pub struct CompiledGCompAte;

impl PosteriorFunctionalEvaluator for GCompAteEvaluator {
    type Compiled = CompiledGCompAte;

    fn compile(&self) -> Result<Self::Compiled, EstimationError> {
        if self.treatment_col >= self.ncols {
            return Err(EstimationError::Stats("treatment column out of range".into()));
        }
        Ok(CompiledGCompAte)
    }

    fn evaluate_batch(
        &self,
        _compiled: &Self::Compiled,
        posterior: PosteriorBatch<'_>,
        output: &mut EffectBatch,
        workspace: &mut PosteriorEvalWorkspace,
        _ctx: &ExecutionContext,
    ) -> Result<(), EstimationError> {
        let n_draws = posterior.len;
        workspace.prepare(n_draws, self.ncols);
        output.prepare(n_draws);

        // Coefficient columns 0..ncols from the batch (ignore extra quantities).
        let mut coef_cols: Vec<&[f64]> = Vec::with_capacity(self.ncols);
        for c in 0..self.ncols {
            let col = posterior.column(c).map_err(|e| EstimationError::Stats(e.to_string()))?;
            coef_cols.push(col);
        }

        for d in 0..n_draws {
            for c in 0..self.ncols {
                workspace.row[c] = coef_cols[c][d];
            }
            let beta = &workspace.row[..self.ncols];
            let mut sum = 0.0;
            for r in 0..self.nrows {
                let mu_a = predict_row(
                    self.family,
                    &self.matrix,
                    self.nrows,
                    self.ncols,
                    self.treatment_col,
                    beta,
                    r,
                    self.active,
                );
                let mu_c = predict_row(
                    self.family,
                    &self.matrix,
                    self.nrows,
                    self.ncols,
                    self.treatment_col,
                    beta,
                    r,
                    self.control,
                );
                sum += mu_a - mu_c;
            }
            output.values[d] = sum / self.nrows as f64;
        }
        Ok(())
    }
}

fn predict_row(
    family: GlmFamily,
    matrix: &[f64],
    nrows: usize,
    ncols: usize,
    t_col: usize,
    beta: &[f64],
    row: usize,
    t_value: f64,
) -> f64 {
    let mut eta = 0.0;
    for c in 0..ncols {
        let x = if c == t_col { t_value } else { matrix[c * nrows + row] };
        eta += x * beta[c];
    }
    family.mean_from_eta(eta)
}

fn likelihood_to_glm_family(l: BayesLikelihood) -> GlmFamily {
    match l {
        BayesLikelihood::GaussianIdentity => GlmFamily::GaussianIdentity,
        BayesLikelihood::BernoulliLogit => GlmFamily::BinomialLogit,
        BayesLikelihood::BernoulliProbit => GlmFamily::BinomialProbit,
        BayesLikelihood::PoissonLog => GlmFamily::PoissonLog,
    }
}

fn prob_err(e: causal_prob::ProbError) -> EstimationError {
    EstimationError::Stats(e.to_string())
}

/// Build a non-identified posterior artifact that still records priors (exit criterion #2).
///
/// Does not invent identification: status remains [`IdentificationStatus::NotIdentified`].
#[must_use]
pub fn nonidentified_with_prior(
    prior: &PriorSet,
    diagnostics: InferenceDiagnostics,
) -> CausalPosterior {
    let mut assumptions = AssumptionSet::new();
    for spec in &prior.specs {
        assumptions.push(AssumptionRecord {
            assumption: Assumption::PriorRestriction(spec.as_assumption()),
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Estimation,
            status: AssumptionStatus::Untestable,
        });
    }
    let schema = PosteriorSchema {
        quantities: Arc::from([PosteriorQuantityKind::Effect { name: Arc::from("ate") }]),
    };
    let draws = PosteriorDraws::from_column_major(schema, 0, Arc::<[f64]>::from(Vec::<f64>::new()))
        .expect("empty draws");
    let summaries = draws.summarize();
    CausalPosterior {
        draws,
        summaries,
        identification: IdentificationStatus::NotIdentified,
        prior_sensitivity: None,
        diagnostics,
        assumptions,
        unidentified_mass: 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
    };
    use causal_data::column::{Float64Column, ValidityBitmap};
    use causal_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
    use causal_expr::{ExprId, IdentifiedEstimand};
    use causal_prob::InferenceDiagnostics;

    fn linear_scm_table(n: usize) -> (TabularData, VariableId, VariableId, VariableId) {
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "Z",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "T",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "Y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let z = VariableId::from_raw(0);
        let t = VariableId::from_raw(1);
        let y = VariableId::from_raw(2);
        let mut zv = vec![0.0; n];
        let mut tv = vec![0.0; n];
        let mut yv = vec![0.0; n];
        for i in 0..n {
            zv[i] = (i as f64) * 0.1;
            tv[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
            yv[i] = 2.0 * tv[i] + 0.5 * zv[i];
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(Float64Column::new(z, Arc::from(zv), validity.clone()).unwrap()),
            OwnedColumn::Float64(Float64Column::new(t, Arc::from(tv), validity.clone()).unwrap()),
            OwnedColumn::Float64(Float64Column::new(y, Arc::from(yv), validity).unwrap()),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        (TabularData::new(storage), t, y, z)
    }

    #[test]
    fn bayesian_and_frequentist_share_ate() {
        let n = 80;
        let (data, t, y, z) = linear_scm_table(n);
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from(vec![z]),
            ExprId::from_raw(0),
        );
        let query = AverageEffectQuery::binary_ate(t, y);

        let freq = crate::adjustment::LinearAdjustmentAte {
            bootstrap_replicates: 0,
            ..crate::adjustment::LinearAdjustmentAte::new()
        };
        let prep = freq.prepare(&data, &estimand, &query).unwrap();
        let mut ws = crate::adjustment::EstimationWorkspace::default();
        let freq_est = freq
            .fit(&prep, &mut ws, &ExecutionContext::for_tests(1), AssumptionSet::new())
            .unwrap();

        let bayes = BayesianGComputationAte {
            backend: BayesianBackendKind::ConjugateGaussian,
            n_draws: 400,
            seed: 5,
            prior_scale: 100.0,
            ..BayesianGComputationAte::new()
        };
        let bprep = bayes.prepare(&data, &estimand, &query).unwrap();
        let mut bws = BayesianGCompWorkspace::default();
        let post = bayes
            .fit(
                &bprep,
                IdentificationStatus::NonparametricallyIdentified,
                &mut bws,
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
        let eq = post.effect_column().unwrap();
        let mean = post.summaries.mean[eq];
        assert!((freq_est.ate - 2.0).abs() < 1e-6, "frequentist ate={}", freq_est.ate);
        assert!((mean - freq_est.ate).abs() < 0.05, "bayes={mean} freq={}", freq_est.ate);
        assert_eq!(post.identification, IdentificationStatus::NonparametricallyIdentified);
    }

    #[test]
    fn prior_does_not_create_identification() {
        let prior = PriorSet::weakly_informative(3);
        let post = nonidentified_with_prior(&prior, InferenceDiagnostics::analytic("none"));
        assert_eq!(post.identification, IdentificationStatus::NotIdentified);
        assert!(!post.assumptions.is_empty());
        assert!((post.unidentified_mass - 1.0).abs() < 1e-12);
    }
}
