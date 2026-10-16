//! Public Study execution for Bayesian quadratic-basis g-computation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, AverageEffectQuery, CausalError, CausalPosterior, CausalQuery, Dag, Diagnostic,
    DiagnosticKind, DiagnosticSeverity, EffectEstimate, EstimatorId, ExecutionContext,
    IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifierId, InferenceMode, Instant,
    OverlapPolicy, PhysicalExecutionPlan, Strategy, Study, StudyResult, TabularData,
    bayesian_draw_count, elapsed_ns, identification_from_cache_or, identify_cached_diagnostic,
    identify_static_query, project_for_ate_estimate, select_claim,
};
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
    ParametricAssumption, PriorAssumption, TargetPopulation,
};
use antecedent_data::TableView;
use antecedent_learn::{BasisTargetPopulation, BayesianBasisGComputation, BayesianBasisSpec};
use antecedent_prob::{PosteriorDraws, PosteriorQuantityKind, PosteriorSchema};

impl Study {
    pub(super) fn execute_bayesian_basis(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let cfg = self.ensure_bayesian_basis_admitted(query)?;
        let (control, active) = binary_levels(query)?;
        let identifier = self.identifier.unwrap_or(IdentifierId::BackdoorAdjustment);
        let (identification, estimand, identify_cached) =
            identification_from_cache_or(ctx, self.identification_cache.as_deref(), || {
                let identification = identify_static_query(
                    identifier,
                    graph,
                    &CausalQuery::AverageEffect(query.clone()),
                )?;
                select_claim(identification, EstimatorId::BayesianBasisGcomp)
            })?;
        let (data_est, query_est, estimand_est) = project_for_ate_estimate(data, query, &estimand)?;
        let data_est = super::super::helpers::apply_scalar_outcome_functional(
            &data_est,
            query_est.outcome,
            &query_est.outcome_functional,
        )?;
        let rows = BasisRows::complete(
            &data_est,
            BasisColumns {
                treatment: query_est.treatment,
                outcome: query_est.outcome,
                adjustment_set: &estimand_est.adjustment_set,
            },
            "bayesian.basis.gcomp row index exceeds row id capacity",
        )?;
        let model = BayesianBasisGComputation;
        let fit = model.fit(
            &rows.as_basis_data(),
            BasisTargetPopulation::AllObserved,
            BayesianBasisSpec {
                prior_sd: cfg.prior_scale,
                n_draws: bayesian_draw_count(&self.inference)?,
                seed: ctx.rng.master_seed(),
            },
            ctx,
        )?;

        let n_draws = fit.ate_draws.len();
        let (posterior, estimate) = basis_posterior(
            &fit,
            &rows,
            &identification,
            BasisEstimatorSettings {
                prior_scale: cfg.prior_scale,
                treatment_contrast: active - control,
                estimator_id: model.estimator_id(),
            },
            "Gaussian identity-link outcome regression with quadratic covariate basis, treatment-by-covariate linear interactions, and a common isotropic zero-centered coefficient prior.",
        )?;
        let mut diagnostics = vec![basis_prediction_diagnostic(&fit)];
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        Ok(self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification,
                estimand_est,
                estimate,
                Strategy::new(identifier, EstimatorId::BayesianBasisGcomp),
                query_est.treatment,
                query_est.outcome,
            )
            .identify_cached(identify_cached)
            .extra_diagnostics(diagnostics)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                posterior: Some(posterior),
                n_draws: Some(u32::try_from(n_draws).unwrap_or(u32::MAX)),
                bootstrap_replicates_requested: Some(None),
                estimate_provenance: Some((
                    Arc::from("estimate.bayesian_basis_gcomp"),
                    Arc::from(model.estimator_id()),
                )),
                ..Default::default()
            }),
        ))
    }

    /// Admission of the quadratic-basis route: all-observed mean contrasts under the
    /// conjugate Gaussian backend with its declared isotropic prior.
    fn ensure_bayesian_basis_admitted(
        &self,
        query: &AverageEffectQuery,
    ) -> Result<&crate::inference::BayesianConfig, CausalError> {
        if query.target_population != TargetPopulation::AllObserved {
            return Err(CausalError::Unsupported {
                message: "bayesian.basis.gcomp supports target_population=AllObserved only",
            });
        }
        let cfg = match &self.inference {
            InferenceMode::Bayesian(cfg) => cfg,
            InferenceMode::Frequentist => {
                return Err(CausalError::Unsupported {
                    message: "bayesian.basis.gcomp requires inference=Bayesian",
                });
            }
        };
        if cfg.likelihood != antecedent_prob::BayesLikelihood::GaussianIdentity
            || cfg.backend != antecedent_estimate::BayesianBackendKind::ConjugateGaussian
            || cfg.has_transfer()
        {
            return Err(CausalError::Unsupported {
                message: "bayesian.basis.gcomp requires the conjugate GaussianIdentity backend with its declared isotropic shrinkage prior; transferred or composed priors are not supported",
            });
        }
        if query.outcome_functional != antecedent_core::OutcomeFunctional::Mean {
            return Err(CausalError::Unsupported {
                message: "bayesian.basis.gcomp currently supports the mean outcome functional only",
            });
        }
        Ok(cfg)
    }
}

/// Prediction summary of a basis fit: draws, rows, targets, range, model and prior ids.
pub(super) fn basis_prediction_diagnostic(
    fit: &antecedent_learn::BayesianBasisEffect,
) -> Diagnostic {
    let n_draws = fit.ate_draws.len();
    Diagnostic::new(
        "estimate.bayesian_basis_gcomp.prediction_diagnostics",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "draws={n_draws}; rows={}; targets={}; min={}; max={}; model={}; prior={}",
            fit.predictions.diagnostics.row_count,
            fit.predictions.diagnostics.target_count,
            fit.predictions.diagnostics.minimum,
            fit.predictions.diagnostics.maximum,
            fit.predictions.provenance.model_id,
            fit.predictions.provenance.prior_id,
        ),
    )
}

/// Posterior draws, summaries, and the effect estimate of a fitted quadratic-basis model:
/// the ATE column followed by one CATE column per complete observed row.
pub(super) fn basis_posterior(
    fit: &antecedent_learn::BayesianBasisEffect,
    rows: &BasisRows,
    identification: &antecedent_identify::IdentificationResult,
    estimator: BasisEstimatorSettings<'_>,
    outcome_model_description: &str,
) -> Result<(CausalPosterior, EffectEstimate), CausalError> {
    let BasisEstimatorSettings { prior_scale, treatment_contrast, estimator_id } = estimator;
    let n_draws = fit.ate_draws.len();
    let row_count = rows.row_ids.len();
    let mut quantities = Vec::with_capacity(row_count + 1);
    quantities.push(PosteriorQuantityKind::Effect { name: Arc::from("ate") });
    quantities.extend((0..row_count).map(|row| PosteriorQuantityKind::Effect {
        name: Arc::from(format!("cate.row.{}", rows.row_ids[row])),
    }));
    // PosteriorPrediction stores [draw,target,row]. Convert to [quantity,draw]
    // while retaining the common draw index across ATE and every CATE row.
    let mut columns = Vec::with_capacity((row_count + 1) * n_draws);
    columns.extend_from_slice(&fit.ate_draws);
    for row in 0..row_count {
        for draw in 0..n_draws {
            let offset = draw * 3 * row_count + 2 * row_count + row;
            columns.push(fit.predictions.values[offset]);
        }
    }
    let draws = PosteriorDraws::from_column_major(
        PosteriorSchema { quantities: Arc::from(quantities) },
        n_draws,
        Arc::<[f64]>::from(columns),
    )?;
    let summaries = draws.summarize();
    let mut assumptions = identification.required_assumptions.clone();
    assumptions.push(AssumptionRecord {
        assumption: Assumption::ParametricRestriction(ParametricAssumption {
            id: Arc::from("bayesian.basis_gcomp.quadratic_outcome"),
            description: Arc::from(outcome_model_description),
        }),
        source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from(estimator_id) },
        scope: AssumptionScope::Estimation,
        status: AssumptionStatus::Declared,
    });
    assumptions.push(AssumptionRecord {
        assumption: Assumption::PriorRestriction(PriorAssumption {
            id: Arc::from(format!("gaussian_coefficients_isotropic_sd_{prior_scale}")),
            description: Arc::from("Every basis coefficient, including the intercept, has a zero-centered Gaussian prior scaled by residual standard deviation."),
        }),
        source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from(estimator_id) },
        scope: AssumptionScope::Estimation,
        status: AssumptionStatus::Declared,
    });
    let posterior = CausalPosterior {
        draws,
        summaries: summaries.clone(),
        identification: identification.status,
        prior_sensitivity: None,
        conflict_summary: None,
        diagnostics: antecedent_prob::InferenceDiagnostics::analytic("conjugate_gaussian"),
        assumptions: assumptions.clone(),
        unidentified_mass: 0.0,
        subsampled_out_mass: 0.0,
        unevaluable_mass: 0.0,
        early_stopped: false,
        treatment_contrast: Some(treatment_contrast),
    };
    let ate_col = 0;
    let ate = summaries.mean[ate_col];
    let se = summaries.sd[ate_col];
    let mut estimate = EffectEstimate::new(ate, se, assumptions, OverlapPolicy::ExplicitOverride)
        .with_n_obs(row_count as u64);
    estimate.heterogeneity.cate =
        Some(Arc::from((0..row_count).map(|row| summaries.mean[row + 1]).collect::<Vec<_>>()));
    Ok((posterior, estimate))
}

/// Complete observed rows of the projected data, in the column layout the basis model takes.
pub(super) struct BasisRows {
    treatment: Vec<f64>,
    outcome: Vec<f64>,
    covariates_colmajor: Vec<f64>,
    n_covariates: usize,
    row_ids: Vec<u32>,
}

/// Prior scale, treatment contrast, and estimator id recorded on a basis posterior.
#[derive(Clone, Copy)]
pub(super) struct BasisEstimatorSettings<'a> {
    pub(super) prior_scale: f64,
    pub(super) treatment_contrast: f64,
    pub(super) estimator_id: &'a str,
}

/// The treatment, outcome, and adjustment columns a basis fit reads.
#[derive(Clone, Copy)]
pub(super) struct BasisColumns<'a> {
    pub(super) treatment: antecedent_core::VariableId,
    pub(super) outcome: antecedent_core::VariableId,
    pub(super) adjustment_set: &'a [antecedent_core::VariableId],
}

impl BasisRows {
    /// Keep the rows where treatment, outcome, and every adjustment covariate are finite.
    pub(super) fn complete(
        data_est: &TabularData,
        columns: BasisColumns<'_>,
        row_id_capacity_message: &'static str,
    ) -> Result<Self, CausalError> {
        let BasisColumns { treatment, outcome, adjustment_set } = columns;
        let treatment = data_est.float64_values(treatment)?;
        let outcome = data_est.float64_values(outcome)?;
        let mut covariates = Vec::with_capacity(adjustment_set.len());
        for variable in adjustment_set.iter().copied() {
            covariates.push(data_est.float64_values(variable)?);
        }
        let n_all = data_est.row_count();
        let rows: Vec<usize> = (0..n_all)
            .filter(|row| {
                treatment[*row].is_finite()
                    && outcome[*row].is_finite()
                    && covariates.iter().all(|column| column[*row].is_finite())
            })
            .collect();
        if rows.len() < 4 {
            return Err(CausalError::Unsupported {
                message: "bayesian.basis.gcomp requires at least four complete observed rows",
            });
        }
        let treatment: Vec<f64> = rows.iter().map(|row| treatment[*row]).collect();
        let outcome: Vec<f64> = rows.iter().map(|row| outcome[*row]).collect();
        let covariates_colmajor: Vec<f64> =
            covariates.iter().flat_map(|column| rows.iter().map(|row| column[*row])).collect();
        let row_ids: Vec<u32> = rows
            .iter()
            .map(|row| {
                u32::try_from(*row)
                    .map_err(|_| CausalError::Compile { message: row_id_capacity_message.into() })
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            treatment,
            outcome,
            covariates_colmajor,
            n_covariates: covariates.len(),
            row_ids,
        })
    }

    pub(super) fn as_basis_data(&self) -> antecedent_learn::BasisData<'_> {
        antecedent_learn::BasisData {
            treatment: &self.treatment,
            outcome: &self.outcome,
            covariates_colmajor: &self.covariates_colmajor,
            n_covariates: self.n_covariates,
            row_ids: Some(&self.row_ids),
            fold_ids: None,
        }
    }
}

fn binary_levels(query: &AverageEffectQuery) -> Result<(f64, f64), CausalError> {
    let level = |intervention: &antecedent_core::Intervention| -> Option<f64> {
        match intervention {
            antecedent_core::Intervention::Set { variable, value }
                if *variable == query.treatment =>
            {
                value.as_f64()
            }
            _ => None,
        }
    };
    let control = level(&query.control);
    let active = level(&query.active);
    match (control, active) {
        (Some(0.0), Some(1.0)) => Ok((0.0, 1.0)),
        _ => Err(CausalError::Unsupported {
            message: "bayesian.basis.gcomp requires treatment interventions Set(T=0) and Set(T=1)",
        }),
    }
}
