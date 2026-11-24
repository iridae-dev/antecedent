//! Posterior sensitivity ranges of a compiled result.
//!
//! * [`SensitivityRangeRequest::EValuePosterior`] and
//!   [`SensitivityRangeRequest::OmittedVariableBiasPosterior`] read the study's own posterior: a
//!   Bayesian average effect fitted under a Gaussian outcome model retains, per draw, the effect
//!   and the residual variance of the outcome model. The E-value draws are the effect over the
//!   residual standard deviation of the same draw ([`evalue_posterior_range`]); the
//!   omitted-variable-bias draws move each effect draw by
//!   `|active - control| sigma_d / sd(T | X) sqrt(R2yz R2dz / (1 - R2dz))`
//!   ([`ovb_posterior_range`]), with `sd(T | X) = sqrt(RSS(T ~ 1 + X) / (n - 2 - k))` on the rows
//!   the linear adjustment reads, the scale the frequentist bias bound uses.
//! * The mediation posterior analyses read the reference posterior of the linear working model
//!   on the retained table ([`MediationReferencePosterior`]): the study's own mediation draws
//!   carry the effects but not the regression summaries the curve is a function of.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    CausalQuery, ExecutionContext, Intervention, OutcomeFunctional, TargetPopulation,
};
use antecedent_kernels::{design_gram_cholesky, sample_gamma, standard_normal};
use antecedent_prob::PosteriorQuantityKind;
use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};
use antecedent_validate::{
    ComponentModel, EffectScale, MediationComponent, MediationPosterior, OvbPosterior,
    ValidationError, component_evalue_posterior_range, component_ovb_posterior_range,
    evalue_posterior_range, mediation_posterior_range, ovb_posterior_range,
};

use super::super::prepared::PreparedStudy;
use super::super::target_validation::result_refutation_target;
use super::{
    Bound, MediationReferencePosterior, SensitivityRangeOutcome, SensitivityRangeRequest,
    complete_cases,
};
use crate::error::CausalError;

/// Stream of the mediator regression's reference posterior.
const MEDIATOR_STREAM: u64 = 0x5345_4E53_4D45_4431;
/// Stream of the outcome regression's reference posterior.
const OUTCOME_STREAM: u64 = 0x5345_4E53_4F55_5431;
/// Most reference posterior draws one analysis takes.
const MAX_REFERENCE_DRAWS: usize = 100_000;

/// The reference posterior of one linear regression: per draw the coefficients and the residual
/// standard deviation.
struct ReferenceDraws {
    coefficients: Vec<Vec<f64>>,
    sigma: Vec<f64>,
}

impl PreparedStudy {
    /// The study's own effect draws and residual variance draws.
    fn effect_and_variance_draws(bound: &Bound<'_>) -> Result<(Vec<f64>, Vec<f64>), CausalError> {
        let posterior = bound.result.posterior.as_ref().ok_or(CausalError::Unsupported {
            message: "a posterior sensitivity range reads the study's posterior; next attempt: \
                      prepare the study with Bayesian inference",
        })?;
        let quantities = &posterior.draws.schema.quantities;
        let effect = posterior
            .effect_column()
            .ok_or(CausalError::Unsupported { message: "the posterior has no effect column" })?;
        let variance = quantities
            .iter()
            .position(|q| matches!(q, PosteriorQuantityKind::ResidualVariance))
            .ok_or(CausalError::Unsupported {
                message: "the posterior retains no residual variance draws; next attempt: a \
                          Bayesian average effect under a Gaussian outcome model",
            })?;
        let column = |q: usize| -> Result<Vec<f64>, CausalError> {
            posterior.draws.column(q).map(<[f64]>::to_vec).map_err(|_| CausalError::Unsupported {
                message: "a posterior column could not be read",
            })
        };
        let (effects, variances) = (column(effect)?, column(variance)?);
        if effects.len() < 2
            || effects.iter().any(|v| !v.is_finite())
            || variances.iter().any(|v| !(v.is_finite() && *v > 0.0))
        {
            return Err(CausalError::Unsupported {
                message: "the posterior needs at least two finite draws with positive residual \
                          variance",
            });
        }
        Ok((effects, variances))
    }

    /// `active - control` of an untransformed average effect without effect modifiers.
    fn average_effect_contrast(&self) -> Result<f64, CausalError> {
        let CausalQuery::AverageEffect(query) = &self.study().query else {
            return Err(CausalError::Unsupported {
                message: "the posterior sensitivity ranges read a Bayesian average effect",
            });
        };
        if !matches!(query.target_population, TargetPopulation::AllObserved)
            || !matches!(query.outcome_functional, OutcomeFunctional::Mean)
            || !query.effect_modifiers.is_empty()
        {
            return Err(CausalError::Unsupported {
                message: "the posterior sensitivity ranges read the mean contrast in all observed \
                          units without effect modifiers",
            });
        }
        let level = |intervention: &Intervention| match intervention {
            Intervention::Set { value, .. } => value.as_f64().filter(|v| v.is_finite()),
            _ => None,
        };
        match (level(&query.control), level(&query.active)) {
            (Some(control), Some(active)) => Ok(active - control),
            _ => Err(CausalError::Unsupported {
                message: "the posterior sensitivity ranges read two set treatment levels",
            }),
        }
    }

    pub(super) fn evalue_posterior(
        &self,
        bound: &Bound<'_>,
        level: f64,
    ) -> Result<SensitivityRangeOutcome, CausalError> {
        self.average_effect_contrast()?;
        let (effects, variances) = Self::effect_and_variance_draws(bound)?;
        let draws: Vec<f64> =
            effects.iter().zip(&variances).map(|(effect, v)| effect / v.sqrt()).collect();
        let (summary, range) = evalue_posterior_range(
            result_refutation_target(bound.result)?,
            std::slice::from_ref(&draws),
            EffectScale::StandardizedMeanDifference,
            level,
        )?;
        Ok(SensitivityRangeOutcome::EValuePosterior { draws, summary, range })
    }

    #[allow(clippy::cast_precision_loss, reason = "the row count is far below 2^53")]
    pub(super) fn ovb_posterior(
        &self,
        bound: &Bound<'_>,
        level: f64,
        r2yz_extreme: f64,
    ) -> Result<SensitivityRangeOutcome, CausalError> {
        let contrast = self.average_effect_contrast()?;
        let (coefficients, variances) = Self::effect_and_variance_draws(bound)?;
        let covariates = bound.covariates();
        let mut ids = vec![bound.outcome, bound.treatment];
        ids.extend(covariates.iter().copied());
        let mut columns = complete_cases(bound.data, &ids)?;
        let adjustment = columns.split_off(2);
        let treatment = &columns[1];
        let n = treatment.len();
        let p = 2 + adjustment.len();
        if n <= p {
            return Err(CausalError::Unsupported {
                message: "the treatment model needs more complete rows than parameters",
            });
        }
        let mut design = vec![1.0; n];
        for column in &adjustment {
            design.extend_from_slice(column);
        }
        let fit = FaerBackend
            .least_squares(&design, n, p - 1, treatment, &mut LeastSquaresWorkspace::default())
            .map_err(ValidationError::from)?;
        let treatment_sd = (fit.rss / (n - p) as f64).sqrt();
        if !(treatment_sd.is_finite() && treatment_sd > 0.0) {
            return Err(CausalError::Unsupported {
                message: "the adjustment set explains the treatment exactly",
            });
        }
        let bias_scale: Vec<f64> =
            variances.iter().map(|v| contrast.abs() * v.sqrt() / treatment_sd).collect();
        let range = ovb_posterior_range(
            result_refutation_target(bound.result)?,
            &OvbPosterior {
                coefficients: std::slice::from_ref(&coefficients),
                bias_scale: std::slice::from_ref(&bias_scale),
            },
            level,
            r2yz_extreme,
        )?;
        Ok(SensitivityRangeOutcome::OmittedVariableBiasPosterior {
            coefficients,
            bias_scale,
            range,
        })
    }

    pub(super) fn mediation_posterior(
        &self,
        bound: &Bound<'_>,
        request: &SensitivityRangeRequest,
        ctx: &ExecutionContext,
    ) -> Result<SensitivityRangeOutcome, CausalError> {
        let draws = match request {
            SensitivityRangeRequest::MediationPosterior { draws, .. }
            | SensitivityRangeRequest::MediationComponentEValuePosterior { draws, .. }
            | SensitivityRangeRequest::MediationComponentOvbPosterior { draws, .. } => *draws,
            _ => {
                return Err(CausalError::Unsupported {
                    message: "not a mediation posterior analysis",
                });
            }
        };
        let (t, m, y, covariates) = self.mediation_columns(bound)?;
        let posterior = reference_mediation_posterior(&t, &m, &y, &covariates, draws, ctx)?;
        let label = |component: MediationComponent| match component {
            MediationComponent::Indirect => "indirect",
            MediationComponent::Direct => "direct",
        };
        let component_draws = |component: MediationComponent| -> Vec<f64> {
            (0..posterior.beta_mediator.len())
                .map(|d| {
                    let indirect = posterior.beta_mediator[d] * posterior.mediator_outcome[d];
                    let value = match component {
                        MediationComponent::Indirect => indirect,
                        MediationComponent::Direct => posterior.total_effect[d] - indirect,
                    };
                    value / posterior.sigma_total[d]
                })
                .collect()
        };
        match request {
            SensitivityRangeRequest::MediationPosterior { component, level, .. } => {
                let range = mediation_posterior_range(
                    bound.coordinate(label(*component))?,
                    &MediationPosterior {
                        beta_mediator: std::slice::from_ref(&posterior.beta_mediator),
                        sigma_total: std::slice::from_ref(&posterior.sigma_total),
                        sigma_mediator: std::slice::from_ref(&posterior.sigma_mediator),
                        rho_tilde: std::slice::from_ref(&posterior.rho_tilde),
                        total_effect: std::slice::from_ref(&posterior.total_effect),
                    },
                    *component,
                    *level,
                )?;
                Ok(SensitivityRangeOutcome::MediationPosterior { posterior, evalue: None, range })
            }
            SensitivityRangeRequest::MediationComponentEValuePosterior {
                component, level, ..
            } => {
                let standardized = component_draws(*component);
                let (summary, range) = component_evalue_posterior_range(
                    bound.coordinate(label(*component))?,
                    std::slice::from_ref(&standardized),
                    EffectScale::StandardizedMeanDifference,
                    *level,
                    ComponentModel::Mediation,
                )?;
                Ok(SensitivityRangeOutcome::MediationPosterior {
                    posterior,
                    evalue: Some(summary),
                    range,
                })
            }
            SensitivityRangeRequest::MediationComponentOvbPosterior {
                level, r2yz_extreme, ..
            } => {
                let bias_scale: Vec<f64> =
                    posterior.sigma_outcome.iter().map(|s| s / posterior.mediator_scale).collect();
                let range = component_ovb_posterior_range(
                    bound.coordinate("mediator_outcome")?,
                    &OvbPosterior {
                        coefficients: std::slice::from_ref(&posterior.mediator_outcome),
                        bias_scale: std::slice::from_ref(&bias_scale),
                    },
                    *level,
                    *r2yz_extreme,
                    ComponentModel::Mediation,
                )?;
                Ok(SensitivityRangeOutcome::MediationPosterior { posterior, evalue: None, range })
            }
            _ => Err(CausalError::Unsupported { message: "not a mediation posterior analysis" }),
        }
    }
}

/// The reference posterior of `y ~ 1 + columns` (`p(beta, sigma^2) ∝ 1 / sigma^2`): per draw
/// `sigma^2 = RSS / chi2_{n - p}` and `beta = beta_hat + sigma L^-T z`, `L L^T = X^T X`, from the
/// keyed stream `stream` of the context seed.
#[allow(clippy::cast_precision_loss, reason = "the row count is far below 2^53")]
fn reference_draws(
    columns: &[&[f64]],
    y: &[f64],
    draws: usize,
    stream: u64,
    ctx: &ExecutionContext,
) -> Result<ReferenceDraws, CausalError> {
    let n = y.len();
    let p = 1 + columns.len();
    if n <= p + 2 {
        return Err(CausalError::Unsupported {
            message: "the working-model posterior needs more complete rows than parameters",
        });
    }
    let mut design = vec![1.0; n];
    for column in columns {
        design.extend_from_slice(column);
    }
    let fit = FaerBackend
        .least_squares(&design, n, p, y, &mut LeastSquaresWorkspace::default())
        .map_err(ValidationError::from)?;
    let l = design_gram_cholesky(&design, n, p).ok_or(CausalError::Unsupported {
        message: "the working-model design is rank deficient",
    })?;
    let dof = (n - p) as f64;
    let mut rng = ctx.rng.stream(stream);
    let mut out = ReferenceDraws {
        coefficients: Vec::with_capacity(draws),
        sigma: Vec::with_capacity(draws),
    };
    for _ in 0..draws {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: "working-model posterior" });
        }
        let chi2 = sample_gamma(dof / 2.0, 0.5, &mut rng);
        let sigma = (fit.rss / chi2.max(f64::MIN_POSITIVE)).sqrt();
        let mut w: Vec<f64> = (0..p).map(|_| standard_normal(&mut rng)).collect();
        for i in (0..p).rev() {
            let mut acc = w[i];
            for q in (i + 1)..p {
                acc -= l[q * p + i] * w[q];
            }
            w[i] = acc / l[i * p + i];
        }
        out.coefficients
            .push(fit.coefficients.iter().zip(&w).map(|(b, w)| b + sigma * w).collect());
        out.sigma.push(sigma);
    }
    Ok(out)
}

/// [`MediationReferencePosterior`] of the working model on complete-case columns.
#[allow(clippy::cast_precision_loss, reason = "the row count is far below 2^53")]
fn reference_mediation_posterior(
    t: &[f64],
    m: &[f64],
    y: &[f64],
    covariates: &[Vec<f64>],
    draws: usize,
    ctx: &ExecutionContext,
) -> Result<MediationReferencePosterior, CausalError> {
    if !(2..=MAX_REFERENCE_DRAWS).contains(&draws) {
        return Err(CausalError::Unsupported {
            message: "the working-model posterior takes between 2 and 100000 draws",
        });
    }
    let mut mediator_columns: Vec<&[f64]> = vec![t];
    mediator_columns.extend(covariates.iter().map(Vec::as_slice));
    let mut outcome_columns: Vec<&[f64]> = vec![t, m];
    outcome_columns.extend(covariates.iter().map(Vec::as_slice));
    let mediator = reference_draws(&mediator_columns, m, draws, MEDIATOR_STREAM, ctx)?;
    let outcome = reference_draws(&outcome_columns, y, draws, OUTCOME_STREAM, ctx)?;
    // The mediator's residual scale on the outcome regression's degrees of freedom.
    let n = y.len();
    let mut design = vec![1.0; n];
    for column in &mediator_columns {
        design.extend_from_slice(column);
    }
    let fit = FaerBackend
        .least_squares(
            &design,
            n,
            mediator_columns.len() + 1,
            m,
            &mut LeastSquaresWorkspace::default(),
        )
        .map_err(ValidationError::from)?;
    let mediator_scale = (fit.rss / (n - outcome_columns.len() - 1) as f64).sqrt();
    if !(mediator_scale.is_finite() && mediator_scale > 0.0) {
        return Err(CausalError::Unsupported {
            message: "the treatment and adjustment set explain the mediator exactly",
        });
    }
    let mut posterior = MediationReferencePosterior {
        beta_mediator: Vec::with_capacity(draws),
        sigma_total: Vec::with_capacity(draws),
        sigma_mediator: Vec::with_capacity(draws),
        rho_tilde: Vec::with_capacity(draws),
        total_effect: Vec::with_capacity(draws),
        mediator_outcome: Vec::with_capacity(draws),
        sigma_outcome: Vec::with_capacity(draws),
        mediator_scale,
    };
    for d in 0..draws {
        let a = mediator.coefficients[d][1];
        let sigma_m = mediator.sigma[d];
        let (c, gamma) = (outcome.coefficients[d][1], outcome.coefficients[d][2]);
        let sigma_y = outcome.sigma[d];
        let sigma_total = (sigma_y * sigma_y + gamma * gamma * sigma_m * sigma_m).sqrt();
        posterior.beta_mediator.push(a);
        posterior.sigma_mediator.push(sigma_m);
        posterior.sigma_total.push(sigma_total);
        posterior.rho_tilde.push(gamma * sigma_m / sigma_total);
        posterior.total_effect.push(c + gamma * a);
        posterior.mediator_outcome.push(gamma);
        posterior.sigma_outcome.push(sigma_y);
    }
    Ok(posterior)
}
