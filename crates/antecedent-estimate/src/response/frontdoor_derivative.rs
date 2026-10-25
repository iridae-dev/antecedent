//! Derivative of a front-door-identified intervention mean of a continuous treatment.
//!
//! With mediators `M` that intercept every causal path from `X` to `Y`, no unblocked back-door
//! path from `X` to `M`, and every back-door path from `M` to `Y` blocked by `X` (and the
//! covariates `Z`), the front-door formula identifies
//! `E[Y | do(x)] = ∫ Σ_{x'} E[Y | m, x', z] P(x' | z) dP(m | x, z) dP(z)`.
//! The estimator fits the outcome as an additive model in `(M, X, Z)`, so the inner average
//! over `x'` leaves `α + Σ_k f_k(m_k) + E g(X) + c(z)`, and the mediator law enters only through
//! the pseudo-outcome
//!
//! `W_i = α̂ + Σ_k f̂_k(M_ik) + mean_j ĝ(X_j) + ĉ(Z_i)`,
//!
//! whose regression on `(X, Z)` is `E[Y | do(x)]` up to `E_Z`. The derivative is the
//! unpenalized-treatment additive-GAM derivative of that second regression (the same plug-in the
//! Jacobian route publishes), so no density of the mediators is modelled and none is assumed
//! Gaussian. The restriction is the additivity of the outcome in the mediators and the treatment.
//!
//! Both stages are refit on every replicate, so the interval carries the first-stage error that
//! a regression on the fixed pseudo-outcome would drop: a whole-procedure row bootstrap
//! (Frequentist, normal interval from the replicate spread) or a Rubin Bayesian bootstrap
//! (Dirichlet(1, …, 1) row weights, equal-tailed credible interval).

use std::sync::Arc;

use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, CausalResponse, CredibleDraws, DerivativeWeighting, Diagnostic,
    DiagnosticKind, DiagnosticSeverity, ExecutionContext, IdentificationStatus,
    ParametricAssumption, ResponseFunctional, ResponseIdentification, ResponseUncertainty,
    ResponseValue, StreamDomain, SupportRegion, SupportReport, SupportStatus, VariableId,
};
use antecedent_data::TabularData;
use antecedent_stats::{AdditiveDesign, GamWorkspace, SmoothSpec, normal_ppf, sample_std};

use crate::EstimationError;

use super::bayes::summarize_scalar_draws;
use super::sample::CompleteSample;
use super::{
    ContinuousResponseEstimator, bootstrap_weights, fit_additive_design, transform_derivative,
};

/// How the front-door derivative reports its uncertainty.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrontDoorDerivativeInference {
    /// Whole-procedure bootstrap with this many replicates; zero publishes the point alone.
    Frequentist {
        /// Replicates.
        replicates: u32,
        /// Circular block length in rows; one resamples rows independently, a longer block
        /// resamples consecutive rows together (overlapping windows of one series).
        block: usize,
    },
    /// Rubin Bayesian bootstrap with this many posterior draws.
    Bayesian {
        /// Draws.
        draws: usize,
        /// Rows sharing one weight, in consecutive blocks; one weights rows independently.
        block: usize,
    },
}

/// Both stages' samples and designs, expanded once and refit under each replicate's weights.
struct Stages {
    first: CompleteSample,
    design_first: AdditiveDesign,
    second: CompleteSample,
    design_second: AdditiveDesign,
}

impl ContinuousResponseEstimator {
    /// The derivative of the front-door-identified `E[Y | do(X = x)]` for a point derivative
    /// (first order, any scale) or an observed-law average derivative, with `mediators` the
    /// front-door set and the estimator's adjustment set the covariates the front-door
    /// formula conditions on.
    ///
    /// # Errors
    ///
    /// A functional other than a first-order point or observed-law average derivative, a
    /// non-point identification status, duplicated or missing columns, a non-converging
    /// additive fit, or fewer than two successful replicates when uncertainty is requested.
    #[allow(
        clippy::too_many_arguments,
        reason = "preserves the existing public signature and its separate scientific, identification and inference inputs"
    )]
    pub fn front_door_derivative(
        &self,
        data: &TabularData,
        functional: &ResponseFunctional,
        mediators: &[VariableId],
        status: IdentificationStatus,
        assumptions: AssumptionSet,
        inference: FrontDoorDerivativeInference,
        ctx: &ExecutionContext,
    ) -> Result<CausalResponse, EstimationError> {
        let (outcome, treatment) = match functional {
            ResponseFunctional::PointDerivative { outcome, treatment, order: 1, .. } => {
                (*outcome, *treatment)
            }
            ResponseFunctional::AverageDerivative {
                outcome,
                treatment,
                weighting: DerivativeWeighting::Observed,
            } => (*outcome, *treatment),
            _ => {
                return Err(EstimationError::unsupported(
                    "the front-door derivative serves a first-order point derivative or an \
                     observed-law average derivative",
                ));
            }
        };
        if mediators.is_empty() {
            return Err(EstimationError::unsupported("a front-door derivative needs mediators"));
        }
        if !matches!(
            status,
            IdentificationStatus::NonparametricallyIdentified
                | IdentificationStatus::IdentifiedUnderParametricRestrictions
        ) {
            return Err(EstimationError::IncompatibleEstimand {
                message: "a front-door derivative requires a point-identified response",
            });
        }
        let stages = self.front_door_stages(data, outcome, treatment, mediators)?;
        let mut workspace = GamWorkspace::default();
        let pseudo = Self::front_door_pseudo_outcome(&stages, None, &mut workspace)?;
        let value = Self::front_door_value(&stages, functional, &pseudo, None, &mut workspace)?;
        let (value, uncertainty, draws_used) = Self::front_door_uncertainty(
            &stages,
            functional,
            inference,
            self.options.confidence_level,
            value,
            ctx,
        )?;
        Ok(Self::front_door_response(
            functional,
            status,
            assumptions,
            &stages,
            value,
            uncertainty,
            draws_used,
        ))
    }

    /// Refit both prepared stages under the declared resampling law, preserving
    /// each replicate's stream and the Bayesian posterior mean publication.
    fn front_door_uncertainty(
        stages: &Stages,
        functional: &ResponseFunctional,
        inference: FrontDoorDerivativeInference,
        level: f64,
        value: f64,
        ctx: &ExecutionContext,
    ) -> Result<(f64, ResponseUncertainty, usize), EstimationError> {
        let n = stages.first.len();
        let (uncertainty, draws_used) = match inference {
            FrontDoorDerivativeInference::Frequentist { replicates: 0, .. } => {
                (ResponseUncertainty::None, 0)
            }
            FrontDoorDerivativeInference::Frequentist { replicates, block } => {
                let values = ctx.map_indexed::<Option<f64>, EstimationError, _>(
                    replicates as usize,
                    |replicate, ctx| {
                        if ctx.cancellation.is_cancelled() {
                            return Err(EstimationError::unsupported(
                                "front-door derivative bootstrap cancelled",
                            ));
                        }
                        let mut rng = ctx.rng.stream_for(StreamDomain::Resample, replicate as u64);
                        let counts = circular_block_counts(n, block, &mut rng)?;
                        let mut local = GamWorkspace::default();
                        // A replicate whose fit fails is dropped, not substituted.
                        Ok(Self::front_door_pseudo_outcome(stages, Some(&counts), &mut local)
                            .and_then(|w| {
                                Self::front_door_value(
                                    stages,
                                    functional,
                                    &w,
                                    Some(&counts),
                                    &mut local,
                                )
                            })
                            .ok()
                            .filter(|v| v.is_finite()))
                    },
                )?;
                let kept: Vec<f64> = values.into_iter().flatten().collect();
                if kept.len() < 2 {
                    return Err(EstimationError::unsupported(
                        "fewer than two front-door bootstrap replicates succeeded",
                    ));
                }
                let se = sample_std(&kept);
                let z = normal_ppf(0.5 + level / 2.0);
                (
                    ResponseUncertainty::Scalar {
                        standard_error: se,
                        level,
                        lower: value - z * se,
                        upper: value + z * se,
                        interpretation: antecedent_core::IntervalInterpretation::Confidence,
                        draws: None,
                    },
                    kept.len(),
                )
            }
            FrontDoorDerivativeInference::Bayesian { draws, block } => {
                let draws = crate::require_bayesian_n_draws(draws)?;
                let values = ctx.map_indexed::<f64, EstimationError, _>(draws, |draw, ctx| {
                    if ctx.cancellation.is_cancelled() {
                        return Err(EstimationError::unsupported(
                            "front-door derivative posterior cancelled",
                        ));
                    }
                    let mut rng = ctx
                        .rng
                        .stream_for(StreamDomain::Bayesian, (0xFD00_0001_u64 << 32) | draw as u64);
                    let weights = block_bayesian_weights(n, block, &mut rng);
                    let mut local = GamWorkspace::default();
                    let pseudo =
                        Self::front_door_pseudo_outcome(stages, Some(&weights), &mut local)?;
                    Self::front_door_value(stages, functional, &pseudo, Some(&weights), &mut local)
                })?;
                let (mean, lo, hi, sd) = summarize_scalar_draws(&values, level)?;
                let used = values.len();
                return Ok((
                    mean,
                    ResponseUncertainty::Scalar {
                        standard_error: sd,
                        level,
                        lower: lo,
                        upper: hi,
                        interpretation: antecedent_core::IntervalInterpretation::Credible,
                        draws: Some(CredibleDraws::scalar(values)),
                    },
                    used,
                ));
            }
        };
        Ok((value, uncertainty, draws_used))
    }

    /// Stage one regresses the outcome on `(X, M, Z)`; stage two regresses the pseudo-outcome on
    /// `(X, Z)`. The second sample is built from the first's complete rows.
    fn front_door_stages(
        &self,
        data: &TabularData,
        outcome: VariableId,
        treatment: VariableId,
        mediators: &[VariableId],
    ) -> Result<Stages, EstimationError> {
        let mut treatments = vec![treatment];
        treatments.extend_from_slice(mediators);
        let first = CompleteSample::read(data, outcome, &treatments, &self.adjustment_set)?;
        let n = first.len();
        let rows: Vec<usize> = (0..n).collect();
        let specs: Vec<SmoothSpec> = (0..first.raw_cols)
            .map(|col| {
                SmoothSpec::new(col, self.options.nuisance_basis, self.options.nuisance_lambda)
            })
            .collect();
        let design_first =
            AdditiveDesign::expand(&first.raw_subset(&rows), n, first.raw_cols, &specs)?;
        // The second table holds a placeholder pseudo-outcome; every fit passes its own.
        let adjustment_names: Vec<String> =
            (0..first.adjustment_cols).map(|j| format!("z{j}")).collect();
        let placeholder = vec![0.0; n];
        let mut columns: Vec<(&str, &[f64])> =
            vec![("w", placeholder.as_slice()), ("x", first.treatments.as_slice())];
        for (j, name) in adjustment_names.iter().enumerate() {
            columns.push((name.as_str(), &first.adjustment[j * n..(j + 1) * n]));
        }
        let table = TabularData::from_f64_columns(columns)?;
        let second_adjustment =
            (0..first.adjustment_cols)
                .map(|j| {
                    u32::try_from(j).ok().and_then(|id| id.checked_add(2))
                    .map(VariableId::from_raw)
                    .ok_or_else(|| EstimationError::unsupported(
                        "front-door adjustment columns exceed the variable identifier range",
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
        let second = CompleteSample::read(
            &table,
            VariableId::from_raw(0),
            &[VariableId::from_raw(1)],
            &second_adjustment,
        )?;
        let design_second = self.outcome_target_design(&second)?;
        Ok(Stages { first, design_first, second, design_second })
    }

    /// `W_i = α̂ + Σ_k f̂_k(M_ik) + mean_j ĝ(X_j) + ĉ(Z_i)` under `weights` (row multiplicities or
    /// Bayesian-bootstrap weights), the front-door plug-in of the outcome given the treatment.
    fn front_door_pseudo_outcome(
        stages: &Stages,
        weights: Option<&[f64]>,
        workspace: &mut GamWorkspace,
    ) -> Result<Vec<f64>, EstimationError> {
        let first = &stages.first;
        let n = first.len();
        let fit = fit_additive_design(&stages.design_first, &first.outcome, weights, workspace)?;
        let smooth = |col: usize| {
            fit.smooth_for_raw_col(col).ok_or_else(|| {
                EstimationError::unsupported("front-door outcome fit is missing a smooth")
            })
        };
        let raw = |col: usize, row: usize| {
            if col < first.treatment_cols {
                first.treatment_matrix[col * n + row]
            } else {
                first.adjustment[(col - first.treatment_cols) * n + row]
            }
        };
        let weight = |row: usize| weights.map_or(1.0, |w| w[row]);
        let total: f64 = (0..n).map(weight).sum();
        let x_smooth = smooth(0)?;
        let mut x_mean = 0.0;
        for row in 0..n {
            x_mean += weight(row) * fit.smooth_partial(x_smooth, raw(0, row))?;
        }
        x_mean /= total;
        let mut pseudo = Vec::with_capacity(n);
        for row in 0..n {
            let mut value = fit.intercept + x_mean;
            for col in 1..first.raw_cols {
                value += fit.smooth_partial(smooth(col)?, raw(col, row))?;
            }
            pseudo.push(value);
        }
        Ok(pseudo)
    }

    /// The derivative of the pseudo-outcome's regression on `(X, Z)` on the functional's scale.
    fn front_door_value(
        stages: &Stages,
        functional: &ResponseFunctional,
        pseudo: &[f64],
        weights: Option<&[f64]>,
        workspace: &mut GamWorkspace,
    ) -> Result<f64, EstimationError> {
        let second = &stages.second;
        let fit = fit_additive_design(&stages.design_second, pseudo, weights, workspace)?;
        if let ResponseFunctional::PointDerivative { at, scale, .. } = functional {
            let (level, gradient) = Self::plugin_gradient_at_fit(&fit, second, &[*at], weights)?;
            transform_derivative(gradient[0], *at, level, *scale)
        } else {
            let smooth = fit.smooth_for_raw_col(0).ok_or_else(|| {
                EstimationError::unsupported("front-door treatment smooth is missing")
            })?;
            let n = second.len();
            let mut total = 0.0;
            let mut mass = 0.0;
            for row in 0..n {
                let w = weights.map_or(1.0, |w| w[row]);
                total += w * fit.smooth_derivative(smooth, second.treatments[row])?;
                mass += w;
            }
            Ok(total / mass)
        }
    }

    fn front_door_response(
        functional: &ResponseFunctional,
        status: IdentificationStatus,
        mut assumptions: AssumptionSet,
        stages: &Stages,
        value: f64,
        uncertainty: ResponseUncertainty,
        replicates: usize,
    ) -> CausalResponse {
        let (minimum, maximum) = crate::util::range(&stages.first.treatments);
        let (status_label, region) = match functional {
            ResponseFunctional::PointDerivative { at, .. } => (
                if (minimum..=maximum).contains(at) {
                    SupportStatus::Supported
                } else {
                    SupportStatus::OutsideEmpiricalSupport
                },
                SupportRegion { minima: Arc::from([*at]), maxima: Arc::from([*at]) },
            ),
            _ => (
                SupportStatus::Supported,
                SupportRegion { minima: Arc::from([minimum]), maxima: Arc::from([maximum]) },
            ),
        };
        let warnings = vec![Diagnostic::new(
            "response.front_door_derivative",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "the derivative of a front-door-identified response: the outcome is fitted as an \
                 additive model in the treatment, the mediators and the covariates, the mediator \
                 terms form the pseudo-outcome whose regression on the treatment is the \
                 intervention mean, and the interval refits both stages on each of {replicates} \
                 replicates (a Frequentist row bootstrap or a Bayesian bootstrap); no mediator \
                 density is modelled, the restriction is additivity of the outcome in the \
                 mediators and treatment"
            ),
        )];
        assumptions.push(AssumptionRecord {
            assumption: Assumption::ParametricRestriction(ParametricAssumption {
                id: Arc::from("response.front_door_derivative.additive_outcome"),
                description: Arc::from(
                    "the outcome regression is additive in the treatment, the front-door \
                     mediators and the covariates, so the front-door average over the \
                     treatment's marginal law separates from the mediator terms",
                ),
            }),
            source: AssumptionSource::AlgorithmDefault {
                algorithm: Arc::from("estimate.response.front_door_derivative"),
            },
            scope: AssumptionScope::Estimation,
            status: AssumptionStatus::Declared,
        });
        CausalResponse {
            estimand: functional.clone(),
            identification_status: status,
            estimate: ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)),
            uncertainty,
            support: SupportReport {
                status: status_label,
                query_region: region,
                diagnostics: Vec::new(),
                warnings,
                point_status: None,
            },
            assumptions,
            provenance_id: Arc::from("estimate.response.front_door_derivative"),
            horizon_identification: None,
            interaction_structurally_zero: false,
        }
    }
}

/// Row multiplicities of a circular block bootstrap: `ceil(n / block)` blocks of `block`
/// consecutive rows (wrapping), started uniformly, rescaled so the multiplicities sum to `n`.
/// A block of one is the independent-row bootstrap.
fn circular_block_counts(
    n: usize,
    block: usize,
    rng: &mut antecedent_core::CausalRng,
) -> Result<Vec<f64>, EstimationError> {
    let block = block.clamp(1, n);
    let blocks = n.div_ceil(block);
    let mut counts = vec![0.0_f64; n];
    for _ in 0..blocks {
        // Multiply-high sampling is strictly below n and therefore fits usize.
        let start = usize::try_from((u128::from(rng.next_u64()) * n as u128) >> 64)
            .map_err(|_| EstimationError::unsupported("front-door block index exceeds usize"))?;
        for offset in 0..block {
            counts[(start + offset) % n] += 1.0;
        }
    }
    let scale = n as f64 / (blocks * block) as f64;
    for count in &mut counts {
        *count *= scale;
    }
    Ok(counts)
}

/// Bayesian-bootstrap weights with one Exp(1) weight per block of consecutive rows, rescaled
/// to mean one. A block of one is the Rubin row-weight posterior.
fn block_bayesian_weights(
    n: usize,
    block: usize,
    rng: &mut antecedent_core::CausalRng,
) -> Vec<f64> {
    let block = block.clamp(1, n);
    if block == 1 {
        return bootstrap_weights(n, rng);
    }
    let blocks = n.div_ceil(block);
    let shared = antecedent_kernels::bayesian_bootstrap_weights(rng, blocks, false);
    let mut weights: Vec<f64> = (0..n).map(|row| shared[row / block]).collect();
    let total: f64 = weights.iter().sum();
    for weight in &mut weights {
        *weight *= n as f64 / total;
    }
    weights
}
