// SPDX-License-Identifier: MIT OR Apache-2.0
//! Local marginal likelihood, cluster sandwich and explicit-prior posterior sampling.
use super::likelihood;
use super::{
    PanelData, PanelDraw, PanelError, PanelFit, PanelFitDiagnostics, PanelFitOptions,
    PanelInference, PanelOptimizerOptions, PanelPrior, check_cancelled, invalid,
};
use antecedent_core::{
    ExecutionContext,
    execution_attempt::{Operation, PhysicalEvent, record_physical, run_operation},
};
use antecedent_kernels::{chol_inverse, cholesky_spd_into, standard_normal};
use antecedent_prob::{all_chains_moved, mcmc_summary_parallel};
use std::sync::Arc;

/// Standard deviation, in whitened posterior units, of the offset each chain starts from.
pub(super) const CHAIN_START_SPREAD: f64 = 2.0;
/// Leapfrog steps per Hamiltonian transition of a panel posterior in whitened coordinates.
pub(super) const PANEL_LEAPFROG_STEPS: u32 = 10;
/// Acceptance probability the panel posterior's step adaptation targets.
pub(super) const PANEL_TARGET_ACCEPT: f64 = 0.8;
/// Most trajectory doublings of a No-U-Turn transition when a posterior is sampled again.
pub(super) const PANEL_NUTS_DEPTH: u32 = 8;
/// Standard deviation, in whitened units, of the offset each chain of a second run starts from.
/// The first run's dispersed starts have already exposed the failure; on a posterior whose
/// Hill shape or carryover decay is flat far from the mass, a start that far out lands on a
/// plateau whose cliffs shrink the adapted step until the chain no longer moves.
pub(super) const RETRY_START_SPREAD: f64 = 0.5;
/// Inverse temperatures of the replica-exchange run, a geometric ladder from the posterior
/// to a copy flattened twentyfold.
pub(super) const PANEL_TEMPERING_LADDER: [f64; 7] = [1.0, 0.6, 0.36, 0.216, 0.13, 0.078, 0.05];

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
/// Central-difference score of an arbitrary objective; the gradient source of every fit
/// whose score has no analytic form.
pub(super) fn optimize_target(
    start: Vec<f64>,
    options: PanelOptimizerOptions,
    ctx: &ExecutionContext,
    objective_fn: &impl Fn(&[f64]) -> f64,
) -> Result<LocalFit, PanelError> {
    optimize_target_with_score(start, options, ctx, objective_fn, &|x| {
        numerical_score(x, ctx, objective_fn)
    })
}
/// A score supplied by a gradient routine, refused when it is not finite exactly as the
/// central-difference score is.
#[cfg(test)]
pub(super) fn checked_score(score: Option<Vec<f64>>) -> Result<Vec<f64>, PanelError> {
    score
        .filter(|score| score.iter().all(|value| value.is_finite()))
        .ok_or_else(|| PanelError::Numerical("nonfinite local likelihood score".into()))
}
/// Score of the marginal log likelihood: exact for independent groups, central differences
/// under a spatial residual law (whose Cholesky coupling has no cheap exact form here).
pub(super) fn log_likelihood_score(
    data: &likelihood::CompiledPanel<'_>,
    x: &[f64],
    ctx: &ExecutionContext,
) -> Result<Vec<f64>, PanelError> {
    check_cancelled(ctx)?;
    if data.family.spatial.is_some() {
        return numerical_score(x, ctx, &|values| likelihood::log_likelihood(data, values));
    }
    let exact: Option<Vec<f64>> =
        super::gradient::independent_score(data, x, &[x.to_vec()], &|_| 0).map(|score| {
            score
                .shared
                .iter()
                .zip(&score.slots[0])
                .map(|(shared, mechanism)| shared + mechanism)
                .collect()
        });
    // A degenerate design (an innovation or intercept scale at its clamp, a perfectly flat
    // outcome) can overflow an exact derivative whose objective is still finite. The
    // central-difference score is the reference behaviour there, so it is used for that
    // evaluation only.
    match exact {
        Some(score) if score.iter().all(|value| value.is_finite()) => Ok(score),
        _ => numerical_score(x, ctx, &|values| likelihood::log_likelihood(data, values)),
    }
}
/// [`optimize_target_with_score`] with an exact score, falling back to the central-difference
/// optimizer when the exact path reaches a point whose score is not finite.
///
/// The exact score moves faster, so on a degenerate design (a flat outcome, a scale at its
/// clamp) it can reach a region the central-difference path never visits. There the fit is
/// rerun exactly as before the exact score existed, so a degenerate design behaves as it did.
pub(super) fn optimize_target_exact(
    start: Vec<f64>,
    options: PanelOptimizerOptions,
    ctx: &ExecutionContext,
    objective_fn: &impl Fn(&[f64]) -> f64,
    score_fn: &impl Fn(&[f64]) -> Result<Vec<f64>, PanelError>,
) -> Result<LocalFit, PanelError> {
    match optimize_target_with_score(start.clone(), options, ctx, objective_fn, score_fn) {
        Err(PanelError::Numerical(detail)) if detail == "nonfinite local likelihood score" => {
            optimize_target(start, options, ctx, objective_fn)
        }
        other => other,
    }
}
/// The shared optimizer; only the source of the score differs between callers.
pub(super) fn optimize_target_with_score(
    start: Vec<f64>,
    options: PanelOptimizerOptions,
    ctx: &ExecutionContext,
    objective_fn: &impl Fn(&[f64]) -> f64,
    score_fn: &impl Fn(&[f64]) -> Result<Vec<f64>, PanelError>,
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
    let mut gradient = score_fn(&values)?;
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
        let next_score = score_fn(&candidate)?;
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

/// Quasi-Newton iterations a frequentist fit runs before an unmet score tolerance switches it
/// to Newton steps on the exact curvature.
///
/// The tolerance is absolute while the likelihood sums every row, so on a large or weakly
/// conditioned panel the quasi-Newton iterate can reach the region where the objective no
/// longer resolves its own gains and then crawl through the whole iteration budget a few
/// multiples above the tolerance, which withholds the sandwich interval. A fit that meets
/// the tolerance within this budget never reaches the Newton phase and is unchanged.
const QUASI_NEWTON_BUDGET: usize = 500;
/// Most Newton steps the polish takes before handing any remaining budget back; every step
/// counts against the declared iteration budget.
const NEWTON_STEPS: usize = 20;

/// The local maximum of a frequentist fit: the quasi-Newton search, then, when it has not met
/// the declared score tolerance, Newton steps on the exact score's own curvature
/// ([`newton_polish`]), then the rest of the declared quasi-Newton budget from wherever the
/// polish stopped. The tolerance keeps its meaning: the fit converges only when the largest
/// absolute score is within it.
fn frequentist_local_fit(
    data: &likelihood::CompiledPanel<'_>,
    start: Vec<f64>,
    options: PanelOptimizerOptions,
    ctx: &ExecutionContext,
) -> Result<LocalFit, PanelError> {
    polished_local_fit(
        start,
        options,
        ctx,
        &|x: &[f64]| likelihood::log_likelihood(data, x),
        &|x: &[f64]| log_likelihood_score(data, x, ctx),
    )
}

/// [`frequentist_local_fit`] for any objective and its score: the quasi-Newton search within
/// [`QUASI_NEWTON_BUDGET`], Newton steps on the central-difference curvature of the score
/// when it has not met the tolerance, then the rest of the declared budget. A search that
/// meets the tolerance within the budget is returned as it is.
pub(super) fn polished_local_fit(
    start: Vec<f64>,
    options: PanelOptimizerOptions,
    ctx: &ExecutionContext,
    objective: &impl Fn(&[f64]) -> f64,
    score: &impl Fn(&[f64]) -> Result<Vec<f64>, PanelError>,
) -> Result<LocalFit, PanelError> {
    let first = PanelOptimizerOptions {
        max_iterations: options.max_iterations.min(QUASI_NEWTON_BUDGET),
        ..options
    };
    let local = optimize_target_exact(start, first, ctx, objective, score)?;
    if local.converged {
        return Ok(local);
    }
    let steps = options.max_iterations.saturating_sub(local.iterations).min(NEWTON_STEPS);
    let polished =
        newton_polish_with(local, options.gradient_tolerance, steps, ctx, objective, score)?;
    let remaining = options.max_iterations.saturating_sub(polished.iterations);
    if polished.converged || remaining == 0 {
        return Ok(polished);
    }
    let rest = PanelOptimizerOptions { max_iterations: remaining, ..options };
    let mut resumed = optimize_target_exact(polished.values.clone(), rest, ctx, objective, score)?;
    resumed.iterations += polished.iterations;
    Ok(if resumed.score_norm <= polished.score_norm { resumed } else { polished })
}

/// Newton steps `x + A⁻¹ g` on the panel log likelihood ([`newton_polish_with`]).
#[cfg(test)]
fn newton_polish(
    data: &likelihood::CompiledPanel<'_>,
    local: LocalFit,
    tolerance: f64,
    steps: usize,
    ctx: &ExecutionContext,
) -> Result<LocalFit, PanelError> {
    newton_polish_with(
        local,
        tolerance,
        steps,
        ctx,
        &|x: &[f64]| likelihood::log_likelihood(data, x),
        &|x: &[f64]| log_likelihood_score(data, x, ctx),
    )
}

/// Newton steps `x + A⁻¹ g` from an iterate that has not met the score tolerance, with `A`
/// the negative Hessian from central differences of the score at the iterate.
///
/// Near the maximum the objective changes by less than its own rounding, so a step is
/// accepted when it lowers the largest absolute score and does not lower the objective by
/// more than a rounding allowance; the backtracking halves the step otherwise. The polish
/// stops at the tolerance, at a curvature that is not positive definite, or when no step
/// improves the score.
fn newton_polish_with(
    mut local: LocalFit,
    tolerance: f64,
    steps: usize,
    ctx: &ExecutionContext,
    objective: &impl Fn(&[f64]) -> f64,
    score: &impl Fn(&[f64]) -> Result<Vec<f64>, PanelError>,
) -> Result<LocalFit, PanelError> {
    let max_abs = |values: &[f64]| values.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
    let mut gradient = score(&local.values)?;
    let mut norm = max_abs(&gradient);
    for _ in 0..steps {
        check_cancelled(ctx)?;
        if norm <= tolerance {
            break;
        }
        let Some(inverse) = score_curvature_inverse(&local.values, ctx, score)? else {
            break;
        };
        let p = local.values.len();
        let direction: Vec<f64> = inverse
            .chunks_exact(p)
            .map(|row| row.iter().zip(&gradient).map(|(a, b)| a * b).sum())
            .collect();
        let allowance = 1e-10 * (1.0 + local.likelihood.abs());
        let mut step = 1.0;
        let mut accepted = None;
        for _ in 0..30 {
            let candidate: Vec<f64> =
                local.values.iter().zip(&direction).map(|(x, d)| x + step * d).collect();
            let value = objective(&candidate);
            if value.is_finite()
                && value >= local.likelihood - allowance
                && let Ok(next) = score(&candidate)
            {
                let next_norm = max_abs(&next);
                if next_norm.is_finite() && next_norm < norm {
                    accepted = Some((candidate, value, next, next_norm));
                    break;
                }
            }
            step *= 0.5;
        }
        let Some((values, value, next, next_norm)) = accepted else {
            break;
        };
        local.values = values;
        local.likelihood = value;
        gradient = next;
        norm = next_norm;
        local.iterations += 1;
    }
    local.score_norm = norm;
    local.converged = norm <= tolerance;
    local.line_search_exhausted = false;
    Ok(local)
}

/// The inverse negative Hessian of an objective from central differences of its score, or
/// `None` when it is not positive definite or a nearby score is not finite.
fn score_curvature_inverse(
    x: &[f64],
    ctx: &ExecutionContext,
    score: &impl Fn(&[f64]) -> Result<Vec<f64>, PanelError>,
) -> Result<Option<Vec<f64>>, PanelError> {
    let p = x.len();
    let mut matrix = vec![0.0; p * p];
    let mut work = x.to_vec();
    for (column, &original) in x.iter().enumerate() {
        check_cancelled(ctx)?;
        let step = 1e-4 * (1.0 + original.abs());
        work[column] = original + step;
        let high = score(&work);
        work[column] = original - step;
        let low = score(&work);
        work[column] = original;
        let (Ok(high), Ok(low)) = (high, low) else {
            return Ok(None);
        };
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
    if matrix.iter().any(|value| !value.is_finite())
        || cholesky_spd_into(&matrix, p, &mut lower).is_err()
    {
        return Ok(None);
    }
    let inverse = chol_inverse(&lower, p);
    Ok(inverse.iter().all(|value| value.is_finite()).then_some(inverse))
}

fn information_inverse(
    data: &likelihood::CompiledPanel<'_>,
    x: &[f64],
    ctx: &ExecutionContext,
) -> Result<Option<Vec<f64>>, PanelError> {
    information_inverse_target(x, ctx, &|values| likelihood::log_likelihood(data, values))
}
pub(super) fn information_inverse_target(
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
        // At a degenerate optimum (a scale at its clamp) the score next to the point can be
        // non-finite: the curvature is then unavailable, never an error of the fit.
        let finite = |result: Result<Vec<f64>, PanelError>| match result {
            Ok(score) => Ok(Some(score)),
            Err(PanelError::Numerical(detail)) if detail == "nonfinite local likelihood score" => {
                Ok(None)
            }
            Err(error) => Err(error),
        };
        work[column] = original + step;
        let Some(high) = finite(numerical_score(&work, ctx, objective))? else {
            return Ok(None);
        };
        work[column] = original - step;
        let Some(low) = finite(numerical_score(&work, ctx, objective))? else {
            return Ok(None);
        };
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
fn marginal_state(
    data: &likelihood::CompiledPanel<'_>,
    x: &[f64],
    detail: likelihood::Detail,
) -> Option<(super::PanelParameters, likelihood::Marginal)> {
    let p = likelihood::decode(data, x)?;
    let residuals = likelihood::residuals(data, &p);
    let marginal = likelihood::marginal(data, &p, &residuals, detail)?;
    Some((p, marginal))
}
fn conditional_draw(
    data: &likelihood::CompiledPanel<'_>,
    x: &[f64],
    rng: &mut antecedent_core::CausalRng,
) -> Option<PanelDraw> {
    let (parameters, marginal) = marginal_state(data, x, likelihood::Detail::Draw)?;
    let group_intercepts = marginal.intercepts(Some(rng))?;
    Some(PanelDraw { parameters, group_intercepts })
}
fn point(data: &likelihood::CompiledPanel<'_>, x: &[f64]) -> Result<PanelDraw, PanelError> {
    let (parameters, marginal) = marginal_state(data, x, likelihood::Detail::Mean)
        .ok_or_else(|| PanelError::Numerical("invalid point mechanism".into()))?;
    let group_intercepts = marginal
        .intercepts(None)
        .ok_or_else(|| PanelError::Numerical("invalid point mechanism".into()))?;
    Ok(PanelDraw { parameters, group_intercepts })
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
    prior.family.validate()
}
/// `log sigmoid(z)`, stable for large `|z|`.
fn log_sigmoid(z: f64) -> f64 {
    -((-z).max(0.0) + (-(-z).abs()).exp().ln_1p())
}
pub(super) fn log_prior(data: &PanelData, x: &[f64], prior: &PanelPrior) -> f64 {
    let Some(p) = likelihood::decode(data, x) else {
        return f64::NEG_INFINITY;
    };
    let layout = likelihood::layout(data);
    let linear = layout.base;
    let mut value =
        -0.5 * x[..linear].iter().map(|value| (value / prior.coefficient_sd).powi(2)).sum::<f64>();
    if data.lag_order == 2 && data.family.is_geometric() {
        // Beta(a, b) on theta = sigmoid(x): the logit Jacobian theta(1-theta) turns the
        // exponents a-1 and b-1 into a and b in unconstrained coordinates.
        let [a, b] = prior.family.geometric_decay;
        for index in 0..data.channels.len() {
            let logit = x[layout.lag + index];
            value += a * log_sigmoid(logit) + b * log_sigmoid(-logit);
        }
    } else if data.lag_order == 2 {
        // The additive-logratio simplex Jacobian is prod(weights). Dirichlet
        // exponent alpha−1 therefore becomes alpha in unconstrained coordinates.
        for (index, channel) in p.channels.iter().enumerate() {
            let weights = channel.lag;
            let alpha = prior.lag.get(index).copied().unwrap_or([1.0; 3]);
            value +=
                weights.iter().zip(alpha).map(|(weight, alpha)| alpha * weight.ln()).sum::<f64>();
        }
    }
    value -= 0.5
        * ((x[layout.k0] - prior.log_half_saturation_mean) / prior.log_half_saturation_sd).powi(2);
    let family = &prior.family;
    for shape in &x[layout.shape..layout.shape + layout.shape_len] {
        value -= 0.5 * ((shape - family.log_hill_shape_mean) / family.log_hill_shape_sd).powi(2);
    }
    // Classes are partially pooled toward the first class on the log scale.
    for class in &x[layout.class..layout.class + layout.class_len] {
        value -= 0.5 * ((class - x[layout.k0]) / family.class_log_saturation_sd).powi(2);
    }
    // Uniform rho on (-1,1), half-Normal sigma/tau. The scale/rho Jacobians
    // are part of the posterior; a prior on log scales would be a different law.
    if data.lag_order == 2 {
        value += (1.0 - p.rho * p.rho).ln();
    }
    value += x[layout.sigma] - 0.5 * (p.innovation_sd / prior.innovation_scale).powi(2);
    value += x[layout.tau] - 0.5 * (p.group_sd / prior.group_scale).powi(2);
    if let Some(range) = layout.range {
        let center =
            family.log_spatial_range_mean.unwrap_or_else(|| data.family.median_distance().ln());
        value -= 0.5 * ((x[range] - center) / family.log_spatial_range_sd).powi(2);
    }
    if value.is_finite() { value } else { f64::NEG_INFINITY }
}

pub(super) fn validate_options(options: &PanelFitOptions) -> Result<usize, PanelError> {
    match &options.inference {
        PanelInference::Frequentist { draws } | PanelInference::FrequentistBootstrap { draws } => {
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

#[allow(
    clippy::too_many_lines,
    reason = "admission, the local fit and every inference policy share one bounded sequence"
)]
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
    let declared = start.clone();
    record_physical(PhysicalEvent::Scan, 1);
    run_operation(Operation::AdjustedFit, || {
        let local = if matches!(options.inference, PanelInference::Frequentist { .. }) {
            frequentist_local_fit(data, start, options.optimizer, ctx)?
        } else {
            let fallback = start.clone();
            match optimize_target_exact(
                start,
                options.optimizer,
                ctx,
                &|x| likelihood::log_likelihood(data, x),
                &|x| log_likelihood_score(data, x, ctx),
            ) {
                Ok(fit) => fit,
                // A posterior only starts and preconditions its sampler from the local fit, and
                // is gated by its own diagnostics: a local search that met a non-finite score
                // starts it from the declared initial point instead of refusing the fit.
                Err(PanelError::Numerical(_))
                    if matches!(options.inference, PanelInference::Bayesian { .. }) =>
                {
                    LocalFit {
                        likelihood: likelihood::log_likelihood(data, &fallback),
                        values: fallback,
                        score_norm: f64::INFINITY,
                        iterations: 0,
                        converged: false,
                        line_search_exhausted: false,
                    }
                }
                Err(error) => return Err(error),
            }
        };
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
            calibration: "not_measured".into(),
        };
        let draws = match &options.inference {
            PanelInference::Frequentist { draws } => {
                frequentist_draws(data, &local, inverse.as_deref(), *draws, &mut diagnostics, ctx)?
            }
            PanelInference::FrequentistBootstrap { draws } => {
                diagnostics.uncertainty = "hierarchical model-based Gaussian/AR parametric bootstrap repeating the original likelihood fitting procedure, plus conditional group-effect draws; shared complete realizations across all reference worlds; not calibrated coverage or a posterior".into();
                if local.converged {
                    bootstrap_draws(data, options, &point, *draws, ctx)?
                } else {
                    vec![None; *draws]
                }
            }
            PanelInference::Bayesian { prior, sampler } => posterior_draws(
                data,
                &local,
                &declared,
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

fn bootstrap_draws(
    data: &likelihood::CompiledPanel<'_>,
    options: &PanelFitOptions,
    fitted: &PanelDraw,
    draws: usize,
    ctx: &ExecutionContext,
) -> Result<Vec<Option<PanelDraw>>, PanelError> {
    let mut output = Vec::with_capacity(draws);
    for replicate in 0..draws {
        check_cancelled(ctx)?;
        let mut rng =
            ctx.rng.stream(antecedent_core::derive_stream(0x5830_3242_4f4f_5453, replicate as u64));
        let mut simulated = data.source.clone();
        let p = &fitted.parameters;
        // Spatially correlated intercepts are drawn jointly before any group's innovations.
        let spatial = likelihood::spatial_prior_intercepts(data, p, &mut rng)
            .ok_or_else(|| PanelError::Numerical("spatial correlation is not factorable".into()))?;
        for (index, group) in simulated.groups.iter_mut().enumerate() {
            let intercept = match &spatial {
                likelihood::SpatialPriorIntercepts::Correlated(draws) => draws[index],
                likelihood::SpatialPriorIntercepts::Independent => {
                    p.group_sd * standard_normal(&mut rng)
                }
            };
            let mut error =
                p.innovation_sd / (1.0 - p.rho.powi(2)).sqrt() * standard_normal(&mut rng);
            for time in 0..group.outcome.len() {
                error = p.rho * error + p.innovation_sd * standard_normal(&mut rng);
                // The process runs through every period; an excluded period stays excluded.
                group.outcome[time] = if data.groups[index].outcome[time].is_nan() {
                    f64::NAN
                } else {
                    likelihood::observed_response(data, p, group, index, time) + intercept + error
                };
            }
        }
        let compiled = likelihood::CompiledPanel::new(&simulated)?;
        // Repeat the declared initialization as well as the optimizer. A warm
        // start at the original optimum would be a different fitted procedure.
        let initial = options.initial.clone().unwrap_or_else(|| likelihood::initial(&compiled));
        let start = likelihood::encode(&compiled, &initial)?;
        let fit = optimize_target_exact(
            start,
            options.optimizer,
            ctx,
            &|x| likelihood::log_likelihood(&compiled, x),
            &|x| log_likelihood_score(&compiled, x, ctx),
        );
        let draw = match fit {
            Ok(fit) if fit.converged => conditional_draw(&compiled, &fit.values, &mut rng),
            Err(PanelError::Cancelled) => return Err(PanelError::Cancelled),
            _ => None,
        };
        output.push(draw);
    }
    Ok(output)
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
        let spatial = data.family.spatial.is_some();
        // A weighted panel is a pseudo-likelihood, so only the group-cluster sandwich is a
        // covariance for it. An unweighted panel is the model's own likelihood, and its
        // inverse observed information is the covariance of the model-conditional claim the
        // fit makes: unlike the sandwich, it does not estimate a variance from a few dozen
        // group score outer products, whose noise left the interval short with 30 to 120
        // groups. Under a spatial residual law the groups are not independent clusters, so
        // the sandwich is not a covariance there either.
        let model_based = spatial || !data.is_weighted();
        if spatial {
            diagnostics.cluster_rank_bound = None;
            diagnostics.uncertainty = "model-based Gaussian global-parameter approximation from the inverse observed information of the spatially dependent likelihood plus joint conditional Gaussian spatial random-effect uncertainty; neither a posterior nor a calibrated interval".into();
        } else if model_based {
            diagnostics.cluster_rank_bound = None;
            diagnostics.uncertainty = "model-based Gaussian global-parameter approximation from the inverse observed information of the panel likelihood plus conditional Gaussian random-effect uncertainty; neither a posterior nor a calibrated interval".into();
        } else {
            diagnostics.cluster_rank_bound =
                Some((data.groups.len() - 1).min(likelihood::parameter_count(data)));
            diagnostics.uncertainty = "local group-cluster Gaussian global-parameter approximation of the weighted pseudo-likelihood plus conditional Gaussian random-effect uncertainty; neither a posterior nor a calibrated interval; the cluster covariance may be rank deficient".into();
        }
        let factor = match inverse.filter(|_| local.converged) {
            Some(inverse) if model_based => model_factor(inverse, local.values.len()),
            Some(inverse) => Some(cluster_factor(data, &local.values, inverse, ctx)?),
            None => None,
        };
        if let Some(factor) = factor {
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
/// Columns of the lower Cholesky factor of a covariance, in the shape of the cluster factor.
fn model_factor(covariance: &[f64], p: usize) -> Option<Vec<Vec<f64>>> {
    let mut lower = vec![0.0; p * p];
    cholesky_spd_into(covariance, p, &mut lower).ok()?;
    Some((0..p).map(|column| (0..p).map(|row| lower[row * p + column]).collect()).collect())
}
/// Stream key of a posterior's second whitened run, so the run after a failed first one does
/// not replay the first one's momenta and starts.
pub(super) const RETRY_STREAM: u64 = 0x5830_3352_4554_5259;

/// The execution context of a posterior's second run: the same budgets, a derived seed.
pub(super) fn retry_context(ctx: &ExecutionContext) -> ExecutionContext {
    retry_context_at(ctx, 0)
}
fn retry_context_at(ctx: &ExecutionContext, attempt: u64) -> ExecutionContext {
    let mut retry = ctx.clone();
    retry.rng = antecedent_core::RngFactory::from_seed(antecedent_core::derive_stream(
        ctx.rng.master_seed(),
        RETRY_STREAM.wrapping_add(attempt),
    ));
    retry
}

/// The center of a second whitening: `start` finished with up to [`NEWTON_STEPS`] Newton steps
/// on the curvature of `score` ([`newton_polish_with`]), or `start` itself when no step
/// improves its score. A center already within `tolerance` is returned as it is.
pub(super) fn polished_center(
    start: &[f64],
    tolerance: f64,
    ctx: &ExecutionContext,
    density: &impl Fn(&[f64]) -> f64,
    score: &impl Fn(&[f64]) -> Result<Vec<f64>, PanelError>,
) -> Result<Vec<f64>, PanelError> {
    let likelihood = density(start);
    if !likelihood.is_finite() {
        return Ok(start.to_vec());
    }
    let local = LocalFit {
        values: start.to_vec(),
        likelihood,
        score_norm: f64::INFINITY,
        iterations: 0,
        converged: false,
        line_search_exhausted: false,
    };
    match newton_polish_with(local, tolerance, NEWTON_STEPS, ctx, density, score) {
        Ok(fit) => Ok(fit.values),
        Err(PanelError::Cancelled) => Err(PanelError::Cancelled),
        Err(_) => Ok(start.to_vec()),
    }
}

/// The covariance of a second whitening at `center`: the inverse curvature of the exact score,
/// else that of the central-difference score of the density, else `None`.
pub(super) fn retry_covariance(
    center: &[f64],
    ctx: &ExecutionContext,
    density: &impl Fn(&[f64]) -> f64,
    score: &impl Fn(&[f64]) -> Result<Vec<f64>, PanelError>,
) -> Result<Option<Vec<f64>>, PanelError> {
    if let Some(inverse) = score_curvature_inverse(center, ctx, score)? {
        return Ok(Some(inverse));
    }
    match information_inverse_target(center, ctx, density) {
        Ok(inverse) => Ok(inverse),
        Err(PanelError::Cancelled) => Err(PanelError::Cancelled),
        Err(_) => Ok(None),
    }
}

/// Sample covariance of retained draws (`p` coordinates each), or `None` when it is not
/// positive definite.
fn draw_covariance(draws: &[f64], p: usize) -> Option<Vec<f64>> {
    let n = draws.len() / p;
    if n < 2 * p {
        return None;
    }
    let mut mean = vec![0.0; p];
    for draw in draws.chunks_exact(p) {
        for (m, x) in mean.iter_mut().zip(draw) {
            *m += x;
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }
    let mut covariance = vec![0.0; p * p];
    for draw in draws.chunks_exact(p) {
        for row in 0..p {
            for column in 0..=row {
                covariance[row * p + column] +=
                    (draw[row] - mean[row]) * (draw[column] - mean[column]);
            }
        }
    }
    for row in 0..p {
        for column in 0..=row {
            let value = covariance[row * p + column] / (n - 1) as f64;
            covariance[row * p + column] = value;
            covariance[column * p + row] = value;
        }
    }
    let mut lower = vec![0.0; p * p];
    (covariance.iter().all(|value| value.is_finite())
        && cholesky_spd_into(&covariance, p, &mut lower).is_ok())
    .then_some(covariance)
}

/// The transition kernel of one whitened run.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Sampler {
    /// Fixed-length Hamiltonian trajectories: every posterior's first run.
    Hamiltonian,
    /// The No-U-Turn sampler: a run after the first misses its diagnostics.
    Nuts,
    /// Replica exchange of No-U-Turn chains: a run after the No-U-Turn run misses too.
    Tempered,
}

/// One whitened Hamiltonian run of a panel posterior.
struct PosteriorRun {
    /// Retained draws in the original coordinates, chain by chain.
    transformed: Vec<f64>,
    summary: antecedent_prob::McmcSummary,
    converged: bool,
}

#[allow(clippy::too_many_arguments)]
fn whitened_run(
    center: &[f64],
    covariance: Option<&[f64]>,
    warmup: usize,
    kind: Sampler,
    spread: f64,
    sampler: &super::PanelSamplingOptions,
    posterior: &(impl Fn(&[f64]) -> f64 + Sync),
    posterior_score: &(impl Fn(&[f64]) -> Result<Vec<f64>, PanelError> + Sync),
    ctx: &ExecutionContext,
) -> Result<PosteriorRun, PanelError> {
    let p = center.len();
    let mut lower = vec![0.0; p * p];
    let preconditioned =
        covariance.is_some_and(|covariance| cholesky_spd_into(covariance, p, &mut lower).is_ok());
    if !preconditioned {
        lower.fill(0.0);
        for j in 0..p {
            lower[j * p + j] = 1.0;
        }
    }
    let transform = |z: &[f64]| -> Vec<f64> {
        center
            .iter()
            .enumerate()
            .map(|(row, value)| {
                value + lower[row * p..(row + 1) * p].iter().zip(z).map(|(l, z)| l * z).sum::<f64>()
            })
            .collect()
    };
    // Hamiltonian Monte Carlo in the whitened coordinates: the gradient is the exact panel
    // score plus the central-difference gradient of the prior, carried to `z` through the
    // whitening factor (`∇_z = Lᵀ ∇_x`). A whitened random walk could not traverse the
    // curved posteriors of short or lagged panels (R-hat to 1.3, bulk ESS to 11 at 8000
    // draws per chain), and a posterior that misses its diagnostics publishes nothing.
    let target = |z: &[f64]| -> Option<(f64, Vec<f64>)> {
        let x = transform(z);
        let value = posterior(&x);
        if !value.is_finite() {
            return None;
        }
        let gradient_x = posterior_score(&x).ok()?;
        let gradient_z = (0..p)
            .map(|column| (0..p).map(|row| lower[row * p + column] * gradient_x[row]).sum())
            .collect();
        Some((value, gradient_z))
    };
    let nuts = antecedent_prob::gradient_hmc::NutsOptions {
        chains: sampler.chains,
        warmup,
        draws: sampler.draws,
        max_depth: PANEL_NUTS_DEPTH,
        initial_step: sampler.proposal_scale,
        target_accept: PANEL_TARGET_ACCEPT,
    };
    let samples = if kind == Sampler::Nuts {
        antecedent_prob::gradient_hmc::sample_nuts_dispersed_parallel(
            &vec![0.0; p],
            spread,
            nuts,
            ctx,
            &target,
        )
    } else if kind == Sampler::Tempered {
        antecedent_prob::gradient_hmc::sample_tempered_nuts_dispersed_parallel(
            &vec![0.0; p],
            spread,
            nuts,
            &PANEL_TEMPERING_LADDER,
            ctx,
            &target,
        )
    } else {
        antecedent_prob::gradient_hmc::sample_gradient_hmc_dispersed_parallel(
            &vec![0.0; p],
            spread,
            antecedent_prob::gradient_hmc::GradientHmcOptions {
                chains: sampler.chains,
                warmup,
                draws: sampler.draws,
                leapfrog_steps: PANEL_LEAPFROG_STEPS,
                initial_step: sampler.proposal_scale,
                target_accept: PANEL_TARGET_ACCEPT,
            },
            ctx,
            &target,
        )
    }
    .map_err(|error| {
        if ctx.cancellation.is_cancelled() {
            PanelError::Cancelled
        } else {
            PanelError::Numerical(error.to_string())
        }
    })?;
    let transformed: Vec<_> = samples.samples.chunks_exact(p).flat_map(transform).collect();
    let summary = mcmc_summary_parallel(&transformed, sampler.chains, sampler.draws, p, ctx)
        .map_err(PanelError::from)?;
    let converged = summary.max_rhat <= sampler.max_rhat
        && summary.min_bulk_ess >= sampler.min_ess
        && summary.min_tail_ess >= sampler.min_ess
        && all_chains_moved(&transformed, sampler.chains, sampler.draws, p);
    Ok(PosteriorRun { transformed, summary, converged })
}

/// A posterior sampled again after its first run missed its diagnostics, or `None` when no
/// later run meets them either; only a run that meets them replaces the first, so a posterior
/// that converges the first time is untouched and every published posterior met its own
/// diagnostics.
///
/// A first run misses for three reasons, and each later run answers one. Its whitening came
/// from a mode search that stalled short of the score tolerance, or from the maximum
/// likelihood fit run off along a direction the likelihood leaves flat (a Hill shape or
/// half-saturation far beyond the doses), where the curvature is a poor scale or not positive
/// definite at all. The second run is therefore whitened at the better, by posterior density,
/// of the modes searched from that fit and from the declared initial point, each finished with
/// Newton steps, by the inverse curvature of the exact score there (else the first run's draw
/// covariance). A fixed-length trajectory crosses a curved ridge (a Hill shape and
/// half-saturation the doses identify only jointly) too slowly, so the second run uses the
/// No-U-Turn sampler with twice the warmup, chains started near that mode. Chains that start
/// beside one another and still disagree sit in separate modes (persistence carried by the
/// autoregression in one and by the group intercepts in the other), so the third run is
/// replica exchange, which carries states between them through tempered copies of the
/// posterior, from dispersed starts.
#[allow(clippy::too_many_arguments)]
fn resample(
    mode: &[f64],
    declared: &[f64],
    first: &PosteriorRun,
    optimizer: PanelOptimizerOptions,
    sampler: &super::PanelSamplingOptions,
    posterior: &(impl Fn(&[f64]) -> f64 + Sync),
    posterior_score: &(impl Fn(&[f64]) -> Result<Vec<f64>, PanelError> + Sync),
    ctx: &ExecutionContext,
) -> Result<Option<PosteriorRun>, PanelError> {
    let p = mode.len();
    let from_declared = match optimize_target(declared.to_vec(), optimizer, ctx, posterior) {
        Ok(fit) => Some(fit.values),
        Err(PanelError::Cancelled) => return Err(PanelError::Cancelled),
        Err(_) => None,
    };
    let mut best: Option<(f64, Vec<f64>)> = None;
    for start in std::iter::once(mode.to_vec()).chain(from_declared) {
        let center =
            polished_center(&start, optimizer.gradient_tolerance, ctx, posterior, posterior_score)?;
        let value = posterior(&center);
        if value.is_finite() && best.as_ref().is_none_or(|(incumbent, _)| value > *incumbent) {
            best = Some((value, center));
        }
    }
    let center = best.map_or_else(|| mode.to_vec(), |(_, center)| center);
    let covariance = match retry_covariance(&center, ctx, posterior, posterior_score)? {
        Some(covariance) => Some(covariance),
        None => draw_covariance(&first.transformed, p),
    };
    let warmup = sampler.warmup.saturating_mul(2);
    let second = whitened_run(
        &center,
        covariance.as_deref(),
        warmup,
        Sampler::Nuts,
        RETRY_START_SPREAD,
        sampler,
        posterior,
        posterior_score,
        &retry_context(ctx),
    )?;
    if second.converged {
        return Ok(Some(second));
    }
    let third = whitened_run(
        &center,
        covariance.as_deref(),
        warmup,
        Sampler::Tempered,
        CHAIN_START_SPREAD,
        sampler,
        posterior,
        posterior_score,
        &retry_context_at(ctx, 1),
    )?;
    Ok(third.converged.then_some(third))
}

#[allow(clippy::too_many_arguments)]
fn posterior_draws(
    data: &likelihood::CompiledPanel<'_>,
    local: &LocalFit,
    declared: &[f64],
    optimizer: PanelOptimizerOptions,
    prior: &PanelPrior,
    sampler: &super::PanelSamplingOptions,
    diagnostics: &mut PanelFitDiagnostics,
    ctx: &ExecutionContext,
) -> Result<Vec<Option<PanelDraw>>, PanelError> {
    let p = likelihood::parameter_count(data);
    let posterior = |x: &[f64]| likelihood::log_likelihood(data, x) + log_prior(data, x, prior);
    // The exact panel score plus the central-difference gradient of the prior.
    let posterior_score = |x: &[f64]| -> Result<Vec<f64>, PanelError> {
        let mut gradient = log_likelihood_score(data, x, ctx)?;
        let mut work = x.to_vec();
        for (j, slot) in gradient.iter_mut().enumerate() {
            let step = 1e-5 * (1.0 + x[j].abs());
            work[j] = x[j] + step;
            let high = log_prior(data, &work, prior);
            work[j] = x[j] - step;
            let low = log_prior(data, &work, prior);
            work[j] = x[j];
            *slot += (high - low) / (2.0 * step);
        }
        Ok(gradient)
    };
    // Priors can materially alter weak likelihood directions. Whitening at the
    // posterior mode uses the exact sampled density, including transform Jacobians.
    // A mode search that meets a non-finite score whitens at the local fit instead, as each
    // temporal specification does; the draws are still gated by their own diagnostics.
    let mode = match optimize_target(local.values.clone(), optimizer, ctx, &posterior) {
        Ok(fit) => fit.values,
        Err(PanelError::Cancelled) => return Err(PanelError::Cancelled),
        Err(_) => local.values.clone(),
    };
    let inverse = match information_inverse_target(&mode, ctx, &posterior) {
        Ok(inverse) => inverse,
        Err(PanelError::Cancelled) => return Err(PanelError::Cancelled),
        Err(_) => None,
    };
    diagnostics.uncertainty = "marginal hierarchical Gaussian-AR1 posterior under explicit priors, with one conditional Gaussian random-intercept draw per group jointly shared across all worlds; no predictive outcome innovations or coverage claim".into();
    let mut run = whitened_run(
        &mode,
        inverse.as_deref(),
        sampler.warmup,
        Sampler::Hamiltonian,
        CHAIN_START_SPREAD,
        sampler,
        &posterior,
        &posterior_score,
        ctx,
    )?;
    if !run.converged
        && let Some(second) =
            resample(&mode, declared, &run, optimizer, sampler, &posterior, &posterior_score, ctx)?
    {
        run = second;
    }
    diagnostics.max_rhat = Some(run.summary.max_rhat);
    diagnostics.min_bulk_ess = Some(run.summary.min_bulk_ess);
    diagnostics.min_tail_ess = Some(run.summary.min_tail_ess);
    diagnostics.posterior_converged = Some(run.converged);
    let transformed = run.transformed;
    // One conditional group-effect draw per retained posterior draw, each on its own
    // stream keyed by its slot, so the draws do not depend on the thread count.
    Ok(ctx.map_indexed(transformed.len() / p, |slot, _inner| {
        check_cancelled(ctx)?;
        let mut rng =
            ctx.rng.stream(antecedent_core::derive_stream(0x5830_3247_524f_5550, slot as u64));
        Ok(conditional_draw(data, &transformed[slot * p..(slot + 1) * p], &mut rng))
    })?)
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

#[cfg(test)]
#[path = "joint_panel_weights_gaps_reference_tests.rs"]
mod joint_panel_weights_gaps_reference_tests;

#[cfg(test)]
mod bootstrap_tests {
    use super::*;
    #[test]
    fn full_procedure_replicates_match_an_independent_static_gaussian_reference() {
        let (mut source, mut truth, _) = super::super::tests::fixture();
        source.lag_order = 0;
        truth.parameters.rho = 0.0;
        for channel in &mut truth.parameters.channels {
            channel.lag = [1.0, 0.0, 0.0];
        }
        let options = PanelFitOptions {
            // A permissive declared stopping rule fixes global coordinates,
            // isolating a closed-form check of simulation and group uncertainty.
            optimizer: PanelOptimizerOptions { max_iterations: 2, gradient_tolerance: 1e100 },
            initial: Some(truth.parameters.clone()),
            inference: PanelInference::FrequentistBootstrap { draws: 3 },
        };
        let ctx = ExecutionContext::for_tests(123);
        let compiled = likelihood::CompiledPanel::new(&source).unwrap();
        let draws = bootstrap_draws(&compiled, &options, &truth, 3, &ctx).unwrap();
        for (replicate, draw) in draws.into_iter().enumerate() {
            let draw = draw.unwrap();
            let decoded_initial = likelihood::decode(
                &compiled,
                &likelihood::encode(&compiled, &truth.parameters).unwrap(),
            )
            .unwrap();
            assert_eq!(draw.parameters, decoded_initial);
            let p = &truth.parameters;
            let mut rng = ctx
                .rng
                .stream(antecedent_core::derive_stream(0x5830_3242_4f4f_5453, replicate as u64));
            let mut expected = Vec::new();
            let mut variances = Vec::new();
            for group in &source.groups {
                let latent = p.group_sd * standard_normal(&mut rng);
                let _initial_error = p.innovation_sd * standard_normal(&mut rng);
                let residual_sum = (0..group.outcome.len())
                    .map(|_| latent + p.innovation_sd * standard_normal(&mut rng))
                    .sum::<f64>();
                let variance = 1.0
                    / (1.0 / p.group_sd.powi(2)
                        + (group.outcome.len() as f64) / p.innovation_sd.powi(2));
                expected.push(variance * residual_sum / p.innovation_sd.powi(2));
                variances.push(variance);
            }
            for ((actual, mean), variance) in
                draw.group_intercepts.iter().zip(expected).zip(variances)
            {
                let reference = mean + variance.sqrt() * standard_normal(&mut rng);
                assert!((actual - reference).abs() < 1e-12);
            }
        }
        let old = PanelFitOptions {
            inference: PanelInference::Frequentist { draws: 3 },
            ..options.clone()
        };
        assert_ne!(options.identity(&source).unwrap(), old.identity(&source).unwrap());
        let fitted = fit(&source, &options, &ctx).unwrap();
        assert!(fitted.draws.iter().all(Option::is_some));
        assert!(fitted.diagnostics.uncertainty.contains("bootstrap"));
    }
}

#[cfg(test)]
mod frequentist_convergence_tests {
    use super::*;

    fn max_abs(values: &[f64]) -> f64 {
        values.iter().map(|value| value.abs()).fold(0.0_f64, f64::max)
    }

    /// From an iterate well short of the maximum, Newton steps on the exact curvature meet a
    /// tolerance far tighter than the one the quasi-Newton search stopped at.
    #[test]
    fn newton_polish_meets_a_tight_tolerance_from_a_stalled_iterate() {
        let (source, _, _) = super::super::tests::fixture();
        let data = likelihood::CompiledPanel::new(&source).unwrap();
        let ctx = ExecutionContext::for_tests(11);
        let start = likelihood::encode(&data, &likelihood::initial(&data)).unwrap();
        let loose = PanelOptimizerOptions { max_iterations: 4_000, gradient_tolerance: 1e-2 };
        let stalled = optimize_target_exact(
            start,
            loose,
            &ctx,
            &|x| likelihood::log_likelihood(&data, x),
            &|x| log_likelihood_score(&data, x, &ctx),
        )
        .unwrap();
        assert!(stalled.converged);
        let tight = 1e-7;
        assert!(
            stalled.score_norm > tight,
            "the loose stop is already tight: {}",
            stalled.score_norm
        );
        let stalled_likelihood = stalled.likelihood;
        let polished = newton_polish(&data, stalled, tight, NEWTON_STEPS, &ctx).unwrap();
        assert!(polished.converged, "score norm {}", polished.score_norm);
        let score = log_likelihood_score(&data, &polished.values, &ctx).unwrap();
        assert!(max_abs(&score) <= tight);
        assert!(
            polished.likelihood >= stalled_likelihood - 1e-10 * (1.0 + stalled_likelihood.abs())
        );
    }

    /// A frequentist fit that meets its tolerance within the quasi-Newton budget is the
    /// quasi-Newton fit itself, bit for bit.
    #[test]
    fn a_fit_converging_within_the_budget_is_unchanged() {
        let (source, _, _) = super::super::tests::fixture();
        let data = likelihood::CompiledPanel::new(&source).unwrap();
        let ctx = ExecutionContext::for_tests(12);
        let start = likelihood::encode(&data, &likelihood::initial(&data)).unwrap();
        let options = PanelOptimizerOptions::default();
        let direct = optimize_target_exact(
            start.clone(),
            options,
            &ctx,
            &|x| likelihood::log_likelihood(&data, x),
            &|x| log_likelihood_score(&data, x, &ctx),
        )
        .unwrap();
        assert!(direct.converged && direct.iterations < QUASI_NEWTON_BUDGET);
        let staged = frequentist_local_fit(&data, start, options, &ctx).unwrap();
        let bits = |values: &[f64]| values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&direct.values), bits(&staged.values));
        assert_eq!(direct.iterations, staged.iterations);
        assert_eq!(direct.likelihood.to_bits(), staged.likelihood.to_bits());
    }
}
