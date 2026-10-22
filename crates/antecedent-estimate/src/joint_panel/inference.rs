// SPDX-License-Identifier: MIT OR Apache-2.0
//! Local marginal likelihood, cluster sandwich and explicit-prior posterior sampling.
use super::likelihood::{self, GaussianGroup};
use super::{
    PanelData, PanelDraw, PanelError, PanelFit, PanelFitDiagnostics, PanelFitOptions,
    PanelInference, PanelOptimizerOptions, PanelPrior, check_cancelled, invalid,
};
use antecedent_core::{
    ExecutionContext,
    execution_attempt::{Operation, PhysicalEvent, record_physical, run_operation},
};
use antecedent_kernels::{chol_inverse, cholesky_spd_into, standard_normal};
use antecedent_prob::{
    all_chains_moved, mcmc_summary,
    metropolis::{MetropolisOptions, sample_metropolis_dispersed},
};
use std::sync::Arc;

/// Standard deviation, in whitened posterior units, of the offset each chain starts from.
pub(super) const CHAIN_START_SPREAD: f64 = 2.0;

fn score(
    data: &likelihood::CompiledPanel<'_>,
    x: &[f64],
    group: Option<usize>,
    ctx: &ExecutionContext,
) -> Result<Vec<f64>, PanelError> {
    let value = |parameters: &[f64]| match group {
        None => likelihood::log_likelihood(data, parameters),
        Some(group) => likelihood::decode(data, parameters)
            .and_then(|p| likelihood::group_likelihood(data, group, &p))
            .map_or(f64::NEG_INFINITY, |value| value.log_likelihood),
    };
    numerical_score(x, ctx, &value)
}
pub(super) fn numerical_score(
    x: &[f64],
    ctx: &ExecutionContext,
    value: &impl Fn(&[f64]) -> f64,
) -> Result<Vec<f64>, PanelError> {
    let mut work = x.to_vec();
    let mut output = Vec::with_capacity(x.len());
    for (coordinate, &original) in x.iter().enumerate() {
        check_cancelled(ctx)?;
        let step = 1e-5 * (1.0 + original.abs());
        work[coordinate] = original + step;
        let high = value(&work);
        work[coordinate] = original - step;
        let low = value(&work);
        work[coordinate] = original;
        let derivative = (high - low) / (2.0 * step);
        if !derivative.is_finite() {
            return Err(PanelError::Numerical("nonfinite local likelihood score".into()));
        }
        output.push(derivative);
    }
    Ok(output)
}

pub(super) struct LocalFit {
    pub(super) values: Vec<f64>,
    pub(super) likelihood: f64,
    pub(super) score_norm: f64,
    pub(super) iterations: usize,
    pub(super) converged: bool,
    pub(super) line_search_exhausted: bool,
}
pub(super) fn optimize_target(
    start: Vec<f64>,
    options: PanelOptimizerOptions,
    ctx: &ExecutionContext,
    objective_fn: &impl Fn(&[f64]) -> f64,
) -> Result<LocalFit, PanelError> {
    if options.max_iterations == 0
        || options.max_iterations > 100_000
        || !options.gradient_tolerance.is_finite()
        || options.gradient_tolerance <= 0.0
    {
        return Err(invalid("optimizer iteration bound or score tolerance"));
    }
    let mut values = start;
    let mut objective = objective_fn(&values);
    if !objective.is_finite() {
        return Err(PanelError::Numerical("invalid starting likelihood".into()));
    }
    let mut gradient = numerical_score(&values, ctx, objective_fn)?;
    let mut curvature = crate::cyclic_optim::InverseCurvature::default();
    let mut iterations = 0;
    let mut exhausted = false;
    for _ in 0..options.max_iterations {
        check_cancelled(ctx)?;
        let norm = gradient.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
        if norm <= options.gradient_tolerance {
            break;
        }
        let (mut direction, _) = curvature.direction(&gradient);
        let maximum = direction.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
        if maximum > 10.0 {
            for value in &mut direction {
                *value *= 10.0 / maximum;
            }
        }
        let slope: f64 = direction.iter().zip(&gradient).map(|(a, b)| a * b).sum();
        let mut step = 1.0;
        let mut accepted = None;
        for _ in 0..40 {
            check_cancelled(ctx)?;
            let candidate: Vec<_> = values
                .iter()
                .zip(&direction)
                .map(|(value, direction)| value + step * direction)
                .collect();
            let likelihood = objective_fn(&candidate);
            if likelihood.is_finite() && likelihood >= objective + 1e-4 * step * slope {
                accepted = Some((candidate, likelihood));
                break;
            }
            step *= 0.5;
        }
        let Some((candidate, next_objective)) = accepted else {
            exhausted = true;
            break;
        };
        let next_score = numerical_score(&candidate, ctx, objective_fn)?;
        curvature.update(&values, &candidate, &gradient, &next_score);
        values = candidate;
        objective = next_objective;
        gradient = next_score;
        iterations += 1;
    }
    let score_norm = gradient.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
    Ok(LocalFit {
        values,
        likelihood: objective,
        score_norm,
        iterations,
        converged: score_norm <= options.gradient_tolerance,
        line_search_exhausted: exhausted,
    })
}

fn information_inverse(
    data: &likelihood::CompiledPanel<'_>,
    x: &[f64],
    ctx: &ExecutionContext,
) -> Result<Option<Vec<f64>>, PanelError> {
    information_inverse_target(x, ctx, &|values| likelihood::log_likelihood(data, values))
}
fn information_inverse_target(
    x: &[f64],
    ctx: &ExecutionContext,
    objective: &impl Fn(&[f64]) -> f64,
) -> Result<Option<Vec<f64>>, PanelError> {
    let p = x.len();
    let mut matrix = vec![0.0; p * p];
    let mut work = x.to_vec();
    for (column, &original) in x.iter().enumerate() {
        check_cancelled(ctx)?;
        let step = 1e-3 * (1.0 + original.abs());
        work[column] = original + step;
        let high = numerical_score(&work, ctx, objective)?;
        work[column] = original - step;
        let low = numerical_score(&work, ctx, objective)?;
        work[column] = original;
        for (row, (&high, &low)) in high.iter().zip(&low).enumerate() {
            matrix[row * p + column] = -(high - low) / (2.0 * step);
        }
    }
    for row in 0..p {
        for column in 0..row {
            let value = 0.5 * (matrix[row * p + column] + matrix[column * p + row]);
            matrix[row * p + column] = value;
            matrix[column * p + row] = value;
        }
    }
    let mut lower = vec![0.0; p * p];
    if cholesky_spd_into(&matrix, p, &mut lower).is_err() {
        return Ok(None);
    }
    let inverse = chol_inverse(&lower, p);
    Ok(inverse.iter().all(|value| value.is_finite()).then_some(inverse))
}

/// An exact rectangular factor of the possibly rank-deficient cluster covariance.
fn cluster_factor(
    data: &likelihood::CompiledPanel<'_>,
    x: &[f64],
    inverse: &[f64],
    ctx: &ExecutionContext,
) -> Result<Vec<Vec<f64>>, PanelError> {
    let groups = data.groups.len();
    let p = x.len();
    let mut scores =
        (0..groups).map(|group| score(data, x, Some(group), ctx)).collect::<Result<Vec<_>, _>>()?;
    let mean: Vec<_> =
        (0..p).map(|j| scores.iter().map(|score| score[j]).sum::<f64>() / groups as f64).collect();
    let correction = (groups as f64 / (groups - 1) as f64).sqrt();
    for score in &mut scores {
        for (value, mean) in score.iter_mut().zip(&mean) {
            *value -= mean;
        }
    }
    Ok(scores
        .into_iter()
        .map(|score| {
            inverse
                .chunks_exact(p)
                .map(|row| correction * row.iter().zip(&score).map(|(a, b)| a * b).sum::<f64>())
                .collect()
        })
        .collect())
}
fn group_states(
    data: &likelihood::CompiledPanel<'_>,
    x: &[f64],
) -> Option<(super::PanelParameters, Vec<GaussianGroup>)> {
    let p = likelihood::decode(data, x)?;
    let states = (0..data.groups.len())
        .map(|group| likelihood::group_likelihood(data, group, &p))
        .collect::<Option<Vec<_>>>()?;
    Some((p, states))
}
fn conditional_draw(
    data: &likelihood::CompiledPanel<'_>,
    x: &[f64],
    rng: &mut antecedent_core::CausalRng,
) -> Option<PanelDraw> {
    let (parameters, groups) = group_states(data, x)?;
    let group_intercepts: Vec<_> = groups
        .into_iter()
        .map(|group| group.intercept_mean + group.intercept_variance.sqrt() * standard_normal(rng))
        .collect();
    group_intercepts
        .iter()
        .all(|value| value.is_finite())
        .then_some(PanelDraw { parameters, group_intercepts })
}
fn point(data: &likelihood::CompiledPanel<'_>, x: &[f64]) -> Result<PanelDraw, PanelError> {
    let (parameters, groups) = group_states(data, x)
        .ok_or_else(|| PanelError::Numerical("invalid point mechanism".into()))?;
    Ok(PanelDraw {
        parameters,
        group_intercepts: groups.into_iter().map(|group| group.intercept_mean).collect(),
    })
}

pub(super) fn validate_prior(prior: &PanelPrior) -> Result<(), PanelError> {
    if !prior.log_half_saturation_mean.is_finite()
        || ![
            prior.coefficient_sd,
            prior.log_half_saturation_sd,
            prior.innovation_scale,
            prior.group_scale,
        ]
        .iter()
        .chain(prior.lag.iter().flatten())
        .all(|value| value.is_finite() && *value > 0.0)
    {
        return Err(invalid(
            "positive finite coefficient, scale, log(k) and Dirichlet prior parameters",
        ));
    }
    Ok(())
}
pub(super) fn log_prior(data: &PanelData, x: &[f64], prior: &PanelPrior) -> f64 {
    let Some(p) = likelihood::decode(data, x) else {
        return f64::NEG_INFINITY;
    };
    let linear = likelihood::linear_count(data);
    let mut value =
        -0.5 * x[..linear].iter().map(|value| (value / prior.coefficient_sd).powi(2)).sum::<f64>();
    let tail = if data.lag_order == 2 { linear + 2 * data.channels.len() } else { linear };
    if data.lag_order == 2 {
        // The additive-logratio simplex Jacobian is prod(weights). Dirichlet
        // exponent alpha−1 therefore becomes alpha in unconstrained coordinates.
        for (index, channel) in p.channels.iter().enumerate() {
            let weights = channel.lag;
            let alpha = prior.lag.get(index).copied().unwrap_or([1.0; 3]);
            value +=
                weights.iter().zip(alpha).map(|(weight, alpha)| alpha * weight.ln()).sum::<f64>();
        }
    }
    value -=
        0.5 * ((x[tail] - prior.log_half_saturation_mean) / prior.log_half_saturation_sd).powi(2);
    // Uniform rho on (-1,1), half-Normal sigma/tau. The scale/rho Jacobians
    // are part of the posterior; a prior on log scales would be a different law.
    if data.lag_order == 2 {
        value += (1.0 - p.rho * p.rho).ln();
    }
    let scale_offset = tail + if data.lag_order == 2 { 2 } else { 1 };
    value += x[scale_offset] - 0.5 * (p.innovation_sd / prior.innovation_scale).powi(2);
    value += x[scale_offset + 1] - 0.5 * (p.group_sd / prior.group_scale).powi(2);
    if value.is_finite() { value } else { f64::NEG_INFINITY }
}

pub(super) fn validate_options(options: &PanelFitOptions) -> Result<usize, PanelError> {
    match &options.inference {
        PanelInference::Frequentist { draws } => {
            if *draws > 100_000 {
                return Err(invalid("at most 100000 frequentist draw slots"));
            }
            Ok(*draws)
        }
        PanelInference::Bayesian { prior, sampler } => {
            validate_prior(prior)?;
            if !(2..=16).contains(&sampler.chains)
                || !(8..=100_000).contains(&sampler.draws)
                || sampler.warmup > 100_000
                || !sampler.proposal_scale.is_finite()
                || sampler.proposal_scale <= 0.0
                || !sampler.max_rhat.is_finite()
                || sampler.max_rhat < 1.0
                || !sampler.min_ess.is_finite()
                || sampler.min_ess <= 0.0
            {
                return Err(invalid("bounded sampler dimensions and positive convergence policy"));
            }
            sampler
                .chains
                .checked_mul(sampler.draws)
                .ok_or_else(|| invalid("posterior draw count overflow"))
        }
    }
}

pub(super) fn fit(
    data: &PanelData,
    options: &PanelFitOptions,
    ctx: &ExecutionContext,
) -> Result<PanelFit, PanelError> {
    super::validate_fit_options(data, options)?;
    check_cancelled(ctx)?;
    let slots = validate_options(options)?;
    let compiled = likelihood::CompiledPanel::new(data)?;
    let data = &compiled;
    let p = likelihood::parameter_count(data);
    let (scratch, retained) = fit_memory(data, slots)?;
    super::check_memory(
        ctx,
        retained
            .checked_add(u64::try_from(scratch).map_err(|_| invalid("workspace size overflow"))?)
            .ok_or_else(|| invalid("fit peak size overflow"))?,
    )?;
    let _scratch = ctx
        .acquire_admission(
            0,
            u64::try_from(scratch).map_err(|_| invalid("workspace size overflow"))?,
        )
        .map_err(|error| PanelError::Resource(error.to_string()))?;
    let retention = Arc::new(
        ctx.acquire_admission(0, retained)
            .map_err(|error| PanelError::Resource(error.to_string()))?,
    );
    let source_identity = data.identity()?;
    let initial = options.initial.clone().unwrap_or_else(|| likelihood::initial(data));
    let start = likelihood::encode(data, &initial)?;
    record_physical(PhysicalEvent::Scan, 1);
    run_operation(Operation::AdjustedFit, || {
        let local = optimize_target(start, options.optimizer, ctx, &|x| {
            likelihood::log_likelihood(data, x)
        })?;
        let inverse = information_inverse(data, &local.values, ctx)?;
        let point = point(data, &local.values)?;
        let mut diagnostics = PanelFitDiagnostics {
            log_likelihood: local.likelihood,
            score_norm: local.score_norm,
            iterations: local.iterations,
            optimizer_converged: local.converged,
            line_search_exhausted: local.line_search_exhausted,
            curvature_available: inverse.is_some(),
            parameter_count: p,
            cluster_rank_bound: None,
            max_rhat: None,
            min_bulk_ess: None,
            min_tail_ess: None,
            posterior_converged: None,
            failed_draws: 0,
            uncertainty: String::new(),
            claim: "model_conditional_mechanism_only".into(),
            calibration: "unmeasured".into(),
        };
        let draws = match &options.inference {
            PanelInference::Frequentist { draws } => {
                frequentist_draws(data, &local, inverse.as_deref(), *draws, &mut diagnostics, ctx)?
            }
            PanelInference::Bayesian { prior, sampler } => posterior_draws(
                data,
                &local,
                options.optimizer,
                prior,
                sampler,
                &mut diagnostics,
                ctx,
            )?,
        };
        diagnostics.failed_draws = draws.iter().filter(|draw| draw.is_none()).count();
        check_cancelled(ctx)?;
        Ok(PanelFit {
            point,
            draws,
            diagnostics,
            source_identity,
            inference: options.inference.clone(),
            retention,
            retained_volume: retained,
        })
    })
}

fn frequentist_draws(
    data: &likelihood::CompiledPanel<'_>,
    local: &LocalFit,
    inverse: Option<&[f64]>,
    draws: usize,
    diagnostics: &mut PanelFitDiagnostics,
    ctx: &ExecutionContext,
) -> Result<Vec<Option<PanelDraw>>, PanelError> {
    Ok({
        diagnostics.cluster_rank_bound =
            Some((data.groups.len() - 1).min(likelihood::parameter_count(data)));
        diagnostics.uncertainty = "local group-cluster Gaussian global-parameter approximation plus conditional Gaussian random-effect uncertainty; neither a posterior nor a calibrated interval; the cluster covariance may be rank deficient".into();
        if let Some(inverse) = inverse.filter(|_| local.converged) {
            let factor = cluster_factor(data, &local.values, inverse, ctx)?;
            let mut output = Vec::with_capacity(draws);
            for slot in 0..draws {
                check_cancelled(ctx)?;
                let mut rng = ctx
                    .rng
                    .stream(antecedent_core::derive_stream(0x5830_3250_414e_454c, slot as u64));
                let mut x = local.values.clone();
                for column in &factor {
                    let z = standard_normal(&mut rng);
                    for (value, factor) in x.iter_mut().zip(column) {
                        *value += factor * z;
                    }
                }
                output.push(conditional_draw(data, &x, &mut rng));
            }
            output
        } else {
            vec![None; draws]
        }
    })
}
fn posterior_draws(
    data: &likelihood::CompiledPanel<'_>,
    local: &LocalFit,
    optimizer: PanelOptimizerOptions,
    prior: &PanelPrior,
    sampler: &super::PanelSamplingOptions,
    diagnostics: &mut PanelFitDiagnostics,
    ctx: &ExecutionContext,
) -> Result<Vec<Option<PanelDraw>>, PanelError> {
    let p = likelihood::parameter_count(data);
    let posterior = |x: &[f64]| likelihood::log_likelihood(data, x) + log_prior(data, x, prior);
    // Priors can materially alter weak likelihood directions. Whitening at the
    // posterior mode uses the exact sampled density, including transform Jacobians.
    let mode = optimize_target(local.values.clone(), optimizer, ctx, &posterior)?;
    let inverse = information_inverse_target(&mode.values, ctx, &posterior)?;
    Ok({
        diagnostics.uncertainty = "marginal hierarchical Gaussian-AR1 posterior under explicit priors, with one conditional Gaussian random-intercept draw per group jointly shared across all worlds; no predictive outcome innovations or coverage claim".into();
        let mut lower = vec![0.0; p * p];
        let preconditioned = inverse
            .as_deref()
            .is_some_and(|inverse| cholesky_spd_into(inverse, p, &mut lower).is_ok());
        if !preconditioned {
            lower.fill(0.0);
            for j in 0..p {
                lower[j * p + j] = 1.0;
            }
        }
        let transform = |z: &[f64]| -> Vec<f64> {
            mode.values
                .iter()
                .enumerate()
                .map(|(row, value)| {
                    value
                        + lower[row * p..(row + 1) * p]
                            .iter()
                            .zip(z)
                            .map(|(l, z)| l * z)
                            .sum::<f64>()
                })
                .collect()
        };
        let samples = sample_metropolis_dispersed(
            &vec![0.0; p],
            CHAIN_START_SPREAD,
            MetropolisOptions {
                chains: sampler.chains,
                warmup: sampler.warmup,
                draws: sampler.draws,
                proposal_scale: sampler.proposal_scale,
            },
            ctx,
            |z| {
                let x = transform(z);
                posterior(&x)
            },
        )
        .map_err(|error| {
            if ctx.cancellation.is_cancelled() {
                PanelError::Cancelled
            } else {
                PanelError::Numerical(error.to_string())
            }
        })?;
        let transformed: Vec<_> = samples.samples.chunks_exact(p).flat_map(transform).collect();
        let summary = mcmc_summary(&transformed, sampler.chains, sampler.draws, p);
        diagnostics.max_rhat = Some(summary.max_rhat);
        diagnostics.min_bulk_ess = Some(summary.min_bulk_ess);
        diagnostics.min_tail_ess = Some(summary.min_tail_ess);
        diagnostics.posterior_converged = Some(
            summary.max_rhat <= sampler.max_rhat
                && summary.min_bulk_ess >= sampler.min_ess
                && summary.min_tail_ess >= sampler.min_ess
                && all_chains_moved(&transformed, sampler.chains, sampler.draws, p),
        );
        let mut output = Vec::with_capacity(sampler.chains * sampler.draws);
        for (slot, x) in transformed.chunks_exact(p).enumerate() {
            check_cancelled(ctx)?;
            let mut rng =
                ctx.rng.stream(antecedent_core::derive_stream(0x5830_3247_524f_5550, slot as u64));
            output.push(conditional_draw(data, x, &mut rng));
        }
        output
    })
}

fn fit_memory(
    data: &likelihood::CompiledPanel<'_>,
    slots: usize,
) -> Result<(usize, u64), PanelError> {
    let p = likelihood::parameter_count(data);
    let rows = data
        .groups
        .iter()
        .try_fold(0_usize, |n, group| n.checked_add(group.outcome.len()))
        .ok_or_else(|| invalid("row count overflow"))?;
    let one_draw = likelihood::parameter_bytes(data)?
        .checked_add(data.groups.len() * 8 + std::mem::size_of::<Option<PanelDraw>>())
        .ok_or_else(|| invalid("retained draw size overflow"))?;
    let retained = slots
        .checked_add(1)
        .and_then(|n| n.checked_mul(one_draw))
        .and_then(|n| n.checked_add(4096 + 32 * p))
        .ok_or_else(|| invalid("retained fit size overflow"))?;
    let scratch = rows
        .checked_mul(32)
        .and_then(|value| value.checked_add(data.workspace_bytes().ok()?))
        .and_then(|value| value.checked_add(data.plan_bytes().ok()?))
        .and_then(|value| value.checked_add(64 * p * p + 32 * p * data.groups.len()))
        .and_then(|value| {
            slots.checked_mul(24 * p + 128).and_then(|draw_scratch| value.checked_add(draw_scratch))
        })
        .ok_or_else(|| invalid("fit workspace size overflow"))?;
    let retained = u64::try_from(retained).map_err(|_| invalid("retained fit size overflow"))?;
    Ok((scratch, retained))
}
