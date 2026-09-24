//! Public Study execution for Bayesian quadratic-basis g-computation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
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
            || cfg.prior.is_some()
            || cfg.prior_artifact.is_some()
            || cfg.external_compose.is_some()
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
        let treatment = data_est.float64_values(query_est.treatment)?;
        let outcome = data_est.float64_values(query_est.outcome)?;
        let mut covariates = Vec::with_capacity(estimand_est.adjustment_set.len());
        for variable in estimand_est.adjustment_set.iter().copied() {
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
                u32::try_from(*row).map_err(|_| CausalError::Compile {
                    message: "bayesian.basis.gcomp row index exceeds row id capacity".into(),
                })
            })
            .collect::<Result<_, _>>()?;
        let model = BayesianBasisGComputation;
        let fit = model
            .fit(
                &treatment,
                &outcome,
                &covariates_colmajor,
                covariates.len(),
                Some(&row_ids),
                None,
                BasisTargetPopulation::AllObserved,
                BayesianBasisSpec {
                    prior_sd: cfg.prior_scale,
                    n_draws: bayesian_draw_count(&self.inference)?,
                    seed: ctx.rng.master_seed(),
                },
                ctx,
            )
            .map_err(|error| CausalError::Compile { message: error.to_string().into() })?;

        let n_draws = fit.ate_draws.len();
        let mut quantities = Vec::with_capacity(rows.len() + 1);
        quantities.push(PosteriorQuantityKind::Effect { name: Arc::from("ate") });
        quantities.extend((0..rows.len()).map(|row| PosteriorQuantityKind::Effect {
            name: Arc::from(format!("cate.row.{}", row_ids[row])),
        }));
        // PosteriorPrediction stores [draw,target,row]. Convert to [quantity,draw]
        // while retaining the common draw index across ATE and every CATE row.
        let mut columns = Vec::with_capacity((rows.len() + 1) * n_draws);
        columns.extend_from_slice(&fit.ate_draws);
        for row in 0..rows.len() {
            for draw in 0..n_draws {
                let offset = draw * 3 * rows.len() + 2 * rows.len() + row;
                columns.push(fit.predictions.values[offset]);
            }
        }
        let draws = PosteriorDraws::from_column_major(
            PosteriorSchema { quantities: Arc::from(quantities) },
            n_draws,
            Arc::<[f64]>::from(columns),
        )
        .map_err(|error| CausalError::Compile { message: error.to_string().into() })?;
        let summaries = draws.summarize();
        let mut assumptions = identification.required_assumptions.clone();
        assumptions.push(AssumptionRecord {
            assumption: Assumption::ParametricRestriction(ParametricAssumption {
                id: Arc::from("bayesian.basis_gcomp.quadratic_outcome"),
                description: Arc::from("Gaussian identity-link outcome regression with quadratic covariate basis, treatment-by-covariate linear interactions, and a common isotropic zero-centered coefficient prior."),
            }),
            source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from(model.estimator_id()) },
            scope: AssumptionScope::Estimation,
            status: AssumptionStatus::Declared,
        });
        assumptions.push(AssumptionRecord {
            assumption: Assumption::PriorRestriction(PriorAssumption {
                id: Arc::from(format!("gaussian_coefficients_isotropic_sd_{}", cfg.prior_scale)),
                description: Arc::from("Every basis coefficient, including the intercept, has a zero-centered Gaussian prior scaled by residual standard deviation."),
            }),
            source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from(model.estimator_id()) },
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
            treatment_contrast: Some(active - control),
        };
        let ate_col = 0;
        let ate = summaries.mean[ate_col];
        let se = summaries.sd[ate_col];
        let mut estimate =
            EffectEstimate::new(ate, se, assumptions, OverlapPolicy::ExplicitOverride)
                .with_n_obs(rows.len() as u64);
        estimate.cate =
            Some(Arc::from((0..rows.len()).map(|row| summaries.mean[row + 1]).collect::<Vec<_>>()));
        let mut diagnostics = vec![Diagnostic::new(
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
        )];
        if identify_cached {
            diagnostics.push(identify_cached_diagnostic());
        }
        Ok(self.finish_identified_execute(IdentifiedExecuteFinish {
            physical,
            identification,
            estimand: estimand_est,
            estimate,
            identifier_id: identifier,
            estimator_id: EstimatorId::BayesianBasisGcomp,
            treatment: query_est.treatment,
            outcome: query_est.outcome,
            identify_cached,
            extra_diagnostics: diagnostics,
            refutations: Vec::new(),
            distribution: None,
            mediation: None,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            bootstrap_replicates_ok: None,
            cancelled: false,
            early_stopped: false,
            extras: IdentifiedExecuteExtras {
                posterior: Some(posterior),
                n_draws: Some(u32::try_from(n_draws).unwrap_or(u32::MAX)),
                bootstrap_replicates_requested: Some(None),
                estimate_provenance: Some((
                    Arc::from("estimate.bayesian_basis_gcomp"),
                    Arc::from(model.estimator_id()),
                )),
                ..Default::default()
            },
        }))
    }
}

fn binary_levels(query: &AverageEffectQuery) -> Result<(f64, f64), CausalError> {
    let level = |intervention: &antecedent_core::Intervention| -> Option<f64> {
        match intervention {
            antecedent_core::Intervention::Set { variable, value }
                if *variable == query.treatment =>
            {
                match value {
                    value => value.as_f64(),
                }
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
