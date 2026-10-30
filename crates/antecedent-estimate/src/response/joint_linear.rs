//! Shared Gaussian-linear response fitting and joint world projection.
//!
//! This extends the linear response design and Bayesian g-computation engines. A fit is
//! shared by all world averages; uncertainty is projected jointly from one coefficient
//! covariance or one aligned coefficient posterior. It is conditional on the retained
//! empirical covariate distribution. This component grants no causal or calibration license.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::{Arc, Mutex};

use antecedent_core::execution::AdmissionLease;
use antecedent_core::execution_attempt::{
    Operation, OperationGuard, PhysicalEvent, record_physical,
};
use antecedent_core::{ExecutionContext, IdentificationStatus, KernelPolicy};
use antecedent_stats::{FaerBackend, LeastSquaresWorkspace, SandwichKind, coefficient_covariance};

use crate::{
    BayesianGCompWorkspace, BayesianGComputationAte, CausalPosterior, EstimationError,
    JointCovariance, LinearContrast, PreparedBayesianProblem,
};

/// Numerical realization retained by a compiled Gaussian joint response.
#[derive(Clone, Debug)]
pub enum JointLinearInference {
    /// One Gaussian least-squares fit and its classical coefficient covariance.
    Frequentist,
    /// One existing Gaussian g-computation coefficient posterior shared across worlds.
    Bayesian(BayesianGComputationAte),
}

/// Failure of shared numerical fitting, projection, admission or cancellation.
#[derive(Debug, thiserror::Error)]
pub enum JointLinearError {
    /// Original estimator or backend refusal.
    #[error(transparent)]
    Estimation(#[from] EstimationError),
    /// A retained numerical dependency or projection is invalid.
    #[error("{0}")]
    Invalid(String),
    /// Admission refused before allocation.
    #[error("{0}")]
    Resource(String),
    /// The caller cancelled before a complete output was produced.
    #[error("joint response execution cancelled")]
    Cancelled,
}

/// Shared numerical program of a Gaussian-linear response on one frozen design.
#[derive(Debug)]
pub struct PreparedJointLinear {
    problem: Arc<PreparedBayesianProblem>,
    inference: JointLinearInference,
    kernel: KernelPolicy,
    fit: Mutex<Option<Arc<JointLinearFit>>>,
}

#[derive(Debug)]
struct JointLinearFit {
    law: JointLinearLaw,
    retention: Arc<AdmissionLease>,
}

#[derive(Debug)]
enum JointLinearLaw {
    Frequentist { coefficients: Vec<f64>, covariance: Vec<f64> },
    Bayesian(Box<CausalPosterior>),
}

/// Joint numerical answer in caller-supplied world order.
#[derive(Clone, Debug)]
pub struct JointLinearEvaluation {
    /// World means, conditional on retained empirical covariates and successful shared draws.
    pub means: Arc<[f64]>,
    /// Full covariance of the world means, including cross-world terms.
    pub covariance: JointCovariance,
    /// Whole functional projection when contrast coefficients were supplied.
    pub contrast: Option<LinearContrast>,
    /// Column-major world-by-draw values; every world uses the same coefficient draw slots.
    pub posterior_draws: Option<Arc<[f64]>>,
    /// One contrast value per shared draw slot.
    pub contrast_draws: Option<Arc<[f64]>>,
    /// Number of shared posterior slots, including failed slots; zero for frequentist output.
    pub draw_count: usize,
    /// Shared failure mask. A failed slot remains in every world and contrast array as NaN.
    pub failed_draws: Arc<[bool]>,
    retention: Arc<AdmissionLease>,
}

impl JointLinearEvaluation {
    /// Admitted bytes retained with this result; not process RSS.
    #[must_use]
    pub fn retained_admitted_bytes(&self) -> u64 {
        self.retention.remainder().bytes()
    }
}

impl PreparedJointLinear {
    /// Retain an existing prepared linear response design without fitting it.
    ///
    /// # Errors
    /// Refuses an empty/invalid design, a non-Gaussian likelihood, or a non-iid problem.
    pub fn new(
        problem: PreparedBayesianProblem,
        inference: JointLinearInference,
        kernel: KernelPolicy,
    ) -> Result<Self, JointLinearError> {
        if problem.design.nrows <= problem.design.ncols
            || problem.design.ncols == 0
            || problem.design.matrix.len()
                != problem.design.nrows.saturating_mul(problem.design.ncols)
            || problem.design.outcome.len() != problem.design.nrows
            || !matches!(problem.serial_dependence, crate::SerialDependence::Iid)
            || problem.unit_ids.is_some()
        {
            return Err(JointLinearError::Invalid(
                "joint Gaussian response requires a valid complete iid linear design".into(),
            ));
        }
        if let JointLinearInference::Bayesian(config) = &inference {
            if config.effective_likelihood() != antecedent_prob::BayesLikelihood::GaussianIdentity {
                return Err(JointLinearError::Invalid(
                    "linear joint response requires the Gaussian identity likelihood".into(),
                ));
            }
        }
        Ok(Self { problem: Arc::new(problem), inference, kernel, fit: Mutex::new(None) })
    }

    /// Check joint empirical assignment support for a source with no adjustment variables.
    ///
    /// With an empty identified adjustment set, the joint assignment distribution has no
    /// conditioning coordinates. Reuses the existing weighted-overlap policy; repeated
    /// assignments are never used to certify a source that requires confounder adjustment.
    /// This is a bounded design scan and performs no fitting.
    #[must_use]
    pub fn unadjusted_world_support(&self, assignments: &[(usize, f64)]) -> bool {
        if !self.problem.adjustment_set.is_empty()
            || assignments.is_empty()
            || assignments
                .iter()
                .any(|(column, level)| *column >= self.problem.design.ncols || !level.is_finite())
        {
            return false;
        }
        let n = self.problem.design.nrows;
        let mut cells = std::collections::BTreeMap::<Vec<u64>, usize>::new();
        for row in 0..n {
            let key = assignments
                .iter()
                .map(|(column, _)| {
                    let value = self.problem.design.matrix[column * n + row];
                    if value == 0.0 { 0 } else { value.to_bits() }
                })
                .collect::<Vec<_>>();
            *cells.entry(key).or_default() += 1;
            // Continuous sources without repeated joint cells are not a discrete support law.
            if cells.len() > 4096 {
                return false;
            }
        }
        let key = assignments
            .iter()
            .map(|(_, value)| if *value == 0.0 { 0 } else { value.to_bits() })
            .collect::<Vec<_>>();
        if !cells.contains_key(&key) {
            return false;
        }
        let effective = cells.values().map(|count| *count as f64).collect::<Vec<_>>();
        let probabilities = effective.iter().map(|count| count / n as f64).collect::<Vec<_>>();
        let minimum = probabilities.iter().copied().fold(f64::INFINITY, f64::min);
        let maximum = probabilities.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let clip = crate::overlap::DEFAULT_PROPENSITY_CLIP;
        let extreme_share =
            if probabilities.iter().any(|p| *p < clip || *p > 1.0 - clip) { 1.0 } else { 0.0 };
        crate::retarget::overlap_gate(
            &effective,
            crate::retarget::MIN_WEIGHTED_ARM_N_EFF,
            Some((minimum, maximum)),
            true,
            extreme_share,
        )
    }

    /// Frozen design and row selection shared by all world projections.
    #[must_use]
    pub fn problem(&self) -> &PreparedBayesianProblem {
        &self.problem
    }
    /// Exact numerical realization, prior, draw count and seed retained at compilation.
    #[must_use]
    pub fn inference(&self) -> &JointLinearInference {
        &self.inference
    }
    /// Kernel policy whose change requires a new numerical program.
    #[must_use]
    pub const fn kernel_policy(&self) -> KernelPolicy {
        self.kernel
    }
    /// Whether a successful shared fit is already retained.
    #[must_use]
    pub fn has_fit(&self) -> bool {
        self.fit.lock().is_ok_and(|fit| fit.is_some())
    }

    /// Evaluate arbitrary world-average design rows and one optional linear contrast.
    ///
    /// # Errors
    /// Shape/finite-value errors, a changed kernel policy, the original fit's failure,
    /// cancellation or resource refusal. No marginal interval subtraction is performed.
    pub fn execute(
        &self,
        world_weights: &[Vec<f64>],
        contrast: Option<&[f64]>,
        ctx: &ExecutionContext,
    ) -> Result<JointLinearEvaluation, JointLinearError> {
        cancelled(ctx)?;
        if ctx.adaptive_draws.enabled {
            return Err(JointLinearError::Invalid("a scalar adaptive draw stop does not certify a joint response; use a fixed aligned draw budget".into()));
        }
        if ctx.kernel_policy != self.kernel {
            return Err(JointLinearError::Invalid(
                "joint response kernel policy changed; recompile the numerical dependency".into(),
            ));
        }
        let p = self.problem.design.ncols;
        let k = world_weights.len();
        if k == 0
            || k > 4096
            || world_weights
                .iter()
                .any(|weights| weights.len() != p || weights.iter().any(|v| !v.is_finite()))
            || contrast
                .is_some_and(|weights| weights.len() != k || weights.iter().any(|v| !v.is_finite()))
        {
            return Err(JointLinearError::Invalid(
                "joint response requires finite aligned world rows and contrast coefficients"
                    .into(),
            ));
        }
        let fit = self.shared_fit(ctx)?;
        let draws = match &fit.law {
            JointLinearLaw::Frequentist { .. } => 0,
            JointLinearLaw::Bayesian(posterior) => posterior.draws.n_draws,
        };
        let bytes = projection_bytes(k, draws)?;
        let retention = Arc::new(admit(ctx, bytes)?);
        let operation = OperationGuard::begin(Operation::Integration);
        let mut evaluation = match &fit.law {
            JointLinearLaw::Frequentist { coefficients, covariance } => project_frequentist(
                world_weights,
                contrast,
                coefficients,
                covariance,
                retention,
                ctx,
            )?,
            JointLinearLaw::Bayesian(posterior) => {
                project_posterior(world_weights, contrast, posterior, retention, ctx)?
            }
        };
        cancelled(ctx)?;
        if let Some(coefficients) = contrast {
            evaluation.contrast =
                Some(contract(&evaluation.means, &evaluation.covariance, coefficients)?);
        }
        record_physical(PhysicalEvent::RetainedSummary, 1);
        record_physical(PhysicalEvent::RetainedBytes, bytes);
        operation.complete();
        Ok(evaluation)
    }

    fn shared_fit(&self, ctx: &ExecutionContext) -> Result<Arc<JointLinearFit>, JointLinearError> {
        let mut slot = self.fit.lock().map_err(|_| {
            JointLinearError::Invalid("joint response fit cache was poisoned".into())
        })?;
        if let Some(fit) = slot.as_ref() {
            fit.retention.observe_current();
            record_physical(PhysicalEvent::CacheHit, 1);
            return Ok(Arc::clone(fit));
        }
        cancelled(ctx)?;
        let p = self.problem.design.ncols;
        let n = self.problem.design.nrows;
        let draw_count = match &self.inference {
            JointLinearInference::Frequentist => 0,
            JointLinearInference::Bayesian(config) => config.n_draws,
        };
        let bytes = n
            .checked_mul(p)
            .and_then(|x| x.checked_add(p.checked_mul(p)?.checked_mul(12)?))
            .and_then(|x| x.checked_add(draw_count.checked_mul(p.checked_add(4)?)?))
            .and_then(|x| x.checked_mul(16))
            .and_then(|x| u64::try_from(x).ok())
            .ok_or_else(|| {
                JointLinearError::Resource("joint response fit quote overflow".into())
            })?;
        let retention = Arc::new(admit(ctx, bytes)?);
        let operation = OperationGuard::begin(Operation::AdjustedFit);
        let law = antecedent_core::numerical_admission::with_context(ctx, || {
            antecedent_core::numerical_admission::with_reservation(Arc::clone(&retention), || {
                match &self.inference {
                    JointLinearInference::Frequentist => {
                        let fit = self
                            .problem
                            .design
                            .fit_ols(&FaerBackend, &mut LeastSquaresWorkspace::default())
                            .map_err(EstimationError::from)?;
                        if fit.rank != p {
                            return Err(JointLinearError::Invalid(
                                "joint response design is rank deficient".into(),
                            ));
                        }
                        let covariance = coefficient_covariance(
                            &self.problem.design.matrix,
                            n,
                            p,
                            &fit.residuals,
                            SandwichKind::Homoskedastic,
                        )
                        .map_err(EstimationError::from)?;
                        Ok(JointLinearLaw::Frequentist {
                            coefficients: fit.coefficients,
                            covariance,
                        })
                    }
                    JointLinearInference::Bayesian(config) => config
                        .fit(
                            &self.problem,
                            IdentificationStatus::NonparametricallyIdentified,
                            &mut BayesianGCompWorkspace::default(),
                            ctx,
                        )
                        .map(|posterior| JointLinearLaw::Bayesian(Box::new(posterior)))
                        .map_err(JointLinearError::from),
                }
            })
        })?;
        cancelled(ctx)?;
        let fit = Arc::new(JointLinearFit { law, retention });
        operation.complete();
        *slot = Some(Arc::clone(&fit));
        Ok(fit)
    }
}

fn project_frequentist(
    weights: &[Vec<f64>],
    contrast: Option<&[f64]>,
    coefficients: &[f64],
    coefficient_covariance: &[f64],
    retention: Arc<AdmissionLease>,
    ctx: &ExecutionContext,
) -> Result<JointLinearEvaluation, JointLinearError> {
    let k = weights.len();
    let p = coefficients.len();
    let means: Vec<_> = weights
        .iter()
        .map(|world| world.iter().zip(coefficients).map(|(x, b)| x * b).sum::<f64>())
        .collect();
    let _scratch = admit(
        ctx,
        u64::try_from(p.checked_mul(8).ok_or_else(|| {
            JointLinearError::Resource("covariance scratch quote overflow".into())
        })?)
        .map_err(|_| JointLinearError::Resource("covariance scratch quote overflow".into()))?,
    )?;
    let mut projected = vec![0.0; p];
    let mut covariance = vec![0.0; k * k];
    for (column, right) in weights.iter().enumerate() {
        cancelled(ctx)?;
        for (index, value) in projected.iter_mut().enumerate() {
            *value = right
                .iter()
                .enumerate()
                .map(|(j, y)| y * coefficient_covariance[index * p + j])
                .sum();
        }
        for (row, left) in weights.iter().enumerate() {
            covariance[column * k + row] = left.iter().zip(&projected).map(|(x, y)| x * y).sum();
        }
    }
    let covariance = JointCovariance { dim: k, values: covariance.into() };
    let scalar = contrast.map(|c| contract(&means, &covariance, c)).transpose()?;
    Ok(JointLinearEvaluation {
        means: means.into(),
        covariance,
        contrast: scalar,
        posterior_draws: None,
        contrast_draws: None,
        draw_count: 0,
        failed_draws: Arc::from([]),
        retention,
    })
}

fn project_posterior(
    weights: &[Vec<f64>],
    contrast: Option<&[f64]>,
    posterior: &CausalPosterior,
    retention: Arc<AdmissionLease>,
    ctx: &ExecutionContext,
) -> Result<JointLinearEvaluation, JointLinearError> {
    let d = posterior.draws.n_draws;
    let k = weights.len();
    let mut values = Vec::with_capacity(k * d);
    for world in weights {
        cancelled(ctx)?;
        values.extend(crate::bayesian::linear_response_draws(posterior, world)?);
    }
    let failed: Vec<_> =
        (0..d).map(|draw| (0..k).any(|world| !values[world * d + draw].is_finite())).collect();
    let successful = failed.iter().filter(|failed| !**failed).count();
    if successful < 2 {
        return Err(JointLinearError::Invalid(
            "joint posterior has fewer than two successful aligned draws".into(),
        ));
    }
    for world in values.chunks_mut(d) {
        for (value, &failed) in world.iter_mut().zip(&failed) {
            if failed {
                *value = f64::NAN;
            }
        }
    }
    let means: Vec<_> = values
        .chunks(d)
        .map(|world| {
            world
                .iter()
                .zip(&failed)
                .filter_map(|(&value, &failed)| (!failed).then_some(value))
                .sum::<f64>()
                / successful as f64
        })
        .collect();
    let mut covariance = vec![0.0; k * k];
    for column in 0..k {
        cancelled(ctx)?;
        for row in 0..k {
            covariance[column * k + row] = (0..d)
                .filter(|&draw| !failed[draw])
                .map(|draw| {
                    (values[row * d + draw] - means[row])
                        * (values[column * d + draw] - means[column])
                })
                .sum::<f64>()
                / (successful - 1) as f64;
        }
    }
    let contrast_draws = contrast.map(|coefficients| {
        (0..d)
            .map(|draw| {
                if failed[draw] {
                    f64::NAN
                } else {
                    coefficients
                        .iter()
                        .enumerate()
                        .map(|(world, coefficient)| coefficient * values[world * d + draw])
                        .sum()
                }
            })
            .collect::<Vec<_>>()
            .into()
    });
    Ok(JointLinearEvaluation {
        means: means.into(),
        covariance: JointCovariance { dim: k, values: covariance.into() },
        contrast: None,
        posterior_draws: Some(values.into()),
        contrast_draws,
        draw_count: d,
        failed_draws: failed.into(),
        retention,
    })
}

fn contract(
    means: &[f64],
    covariance: &JointCovariance,
    coefficients: &[f64],
) -> Result<LinearContrast, JointLinearError> {
    let value = means.iter().zip(coefficients).map(|(x, c)| x * c).sum::<f64>();
    let variance = coefficients
        .iter()
        .enumerate()
        .map(|(i, x)| {
            coefficients.iter().enumerate().map(|(j, y)| x * y * covariance.get(i, j)).sum::<f64>()
        })
        .sum::<f64>();
    let magnitude = coefficients
        .iter()
        .enumerate()
        .map(|(i, x)| {
            coefficients
                .iter()
                .enumerate()
                .map(|(j, y)| (x * y * covariance.get(i, j)).abs())
                .sum::<f64>()
        })
        .sum::<f64>();
    let roundoff = 8.0 * (coefficients.len() as f64 + 1.0) * f64::EPSILON * magnitude;
    if !value.is_finite() || !variance.is_finite() || !magnitude.is_finite() || variance < -roundoff
    {
        return Err(JointLinearError::Invalid(
            "whole-contrast projection is not a finite mean and covariance".into(),
        ));
    }
    Ok(LinearContrast { value, se: variance.max(0.0).sqrt() })
}

fn projection_bytes(worlds: usize, draws: usize) -> Result<u64, JointLinearError> {
    worlds
        .checked_mul(worlds)
        .and_then(|x| x.checked_add(worlds.checked_mul(draws.checked_add(4)?)?))
        .and_then(|x| x.checked_add(draws.checked_mul(2)?))
        .and_then(|x| x.checked_mul(16))
        .and_then(|x| u64::try_from(x).ok())
        .ok_or_else(|| JointLinearError::Resource("joint projection quote overflow".into()))
}
fn cancelled(ctx: &ExecutionContext) -> Result<(), JointLinearError> {
    if ctx.cancellation.is_cancelled() { Err(JointLinearError::Cancelled) } else { Ok(()) }
}
fn admit(ctx: &ExecutionContext, bytes: u64) -> Result<AdmissionLease, JointLinearError> {
    let limit = ctx.memory_limit_bytes();
    if limit.is_some_and(|limit| bytes > limit) {
        return Err(JointLinearError::Resource(
            "joint response exceeds configured memory limit".into(),
        ));
    }
    ctx.acquire_admission(0, bytes).map_err(|error| JointLinearError::Resource(error.to_string()))
}

#[cfg(test)]
mod failed_draw_tests {
    use super::*;
    use antecedent_core::AssumptionSet;
    use antecedent_prob::{InferenceDiagnostics, PosteriorDraws, PosteriorSchema};

    #[test]
    fn joint_summary_is_conditional_on_success_but_retains_shared_failed_slot() {
        let schema = PosteriorSchema::coefficients(2);
        let clean = PosteriorDraws::from_column_major(schema.clone(), 3, vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0]).unwrap();
        let summaries = clean.summarize();
        let draws = PosteriorDraws::from_column_major(schema, 3, vec![1.0, f64::NAN, 3.0, 2.0, f64::NAN, 4.0]).unwrap();
        let posterior = CausalPosterior {
            draws, summaries, identification: IdentificationStatus::NonparametricallyIdentified,
            prior_sensitivity: None, conflict_summary: None,
            diagnostics: InferenceDiagnostics::analytic("test"), assumptions: AssumptionSet::new(),
            unidentified_mass: 0.0, subsampled_out_mass: 0.0, unevaluable_mass: 0.0,
            early_stopped: false, treatment_contrast: None, preparation: None,
        };
        let ctx = ExecutionContext::for_tests(0);
        let result = project_posterior(&[vec![1.0, 0.0], vec![1.0, 1.0]], Some(&[-1.0, 1.0]),
            &posterior, Arc::new(admit(&ctx, 1024).unwrap()), &ctx).unwrap();
        assert_eq!(result.draw_count, 3);
        assert_eq!(result.failed_draws.as_ref(), &[false, true, false]);
        assert_eq!(result.means.as_ref(), &[2.0, 5.0]);
        assert_eq!(result.covariance.values.as_ref(), &[2.0, 4.0, 4.0, 8.0]);
        let worlds = result.posterior_draws.as_ref().unwrap();
        assert_eq!(worlds.len(), 6);
        assert_eq!((worlds[0], worlds[2], worlds[3], worlds[5]), (1.0, 3.0, 3.0, 7.0));
        assert!(worlds[1].is_nan() && worlds[4].is_nan());
        let contrast = result.contrast_draws.as_ref().unwrap();
        assert_eq!(contrast.len(), 3);
        assert_eq!((contrast[0], contrast[2]), (2.0, 4.0));
        assert!(contrast[1].is_nan());
    }
}
