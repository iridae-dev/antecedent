//! Posterior draws of experimentally recovered coefficients.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    AssembledCoefficientRecovery, BlockRank, BlockSolution, BlockSystem, CausalRng,
    CoefficientPrior, ConstraintWeighting, DisturbancePrior, ExperimentalFitError,
    ExperimentalRecoveryFit, IdentifiedPosteriorBlock, POSTERIOR_SWEEP_OPERATION, PosteriorSpec,
    PriorFault, RecoveredCoefficientUncertainty, RecoveredDrawOrigin, RecoveredEstimate,
    RecoveryPrior, VariableId, back_solve_into, chol_inverse, chol_solve_into, cholesky_spd_into,
    diagonal, draw_input, gram_of_upper, keyed_stream, quadratic_n, residual_vector,
    sample_inv_gamma, solve_block, standard_normal,
};

/// One posterior draw of a resolved row.
struct RowDraw {
    coefficients: Vec<f64>,
    intercept: f64,
    variance: f64,
}

/// Draw the recovered law from a Gaussian posterior over the identified subspace.
///
/// The likelihood is the asymptotic Gaussian law of the efficient constraint
/// estimate, `b̂ | b, σ² ~ N(b, σ² B⁻¹)` with `B = Σ_r A_r' n_r Σ_JJ A_r`, a
/// limited-information likelihood of the experimental-effect moments rather than
/// the full cyclic likelihood. The pooled residual scatter informs `σ²`
/// (`N σ̂² | σ² ~ σ² χ²_{N−1}`) and the pooled residual mean informs the
/// intercept under a flat prior (`c | b, σ² ~ N(x̄_i − b'x̄_{-i}, σ²/N)`).
///
/// - [`DisturbancePrior::Known`]: draws are independent; stream
///   [`POSTERIOR_SWEEP_OPERATION`], chain = block, replicate = draw.
/// - [`DisturbancePrior::InverseWishart`]: a Gibbs chain per block alternates
///   `b | σ²` (Gaussian) and `σ² | b` (inverse gamma); `burn_in` sweeps are
///   discarded; replicate = sweep.
///
/// Rows are independent under the diagonal noise law. A rank-deficient block
/// has a posterior over its identified projections only
/// ([`IdentifiedPosteriorBlock`], under a known disturbance variance); its null
/// directions stay unresolved in every draw, whatever the prior. An uncovered
/// row has no posterior.
///
/// # Errors
///
/// [`ExperimentalFitError::DrawCount`], [`ExperimentalFitError::Prior`],
/// [`ExperimentalFitError::PosteriorPrecision`], a block fault, or a law fault.
pub fn posterior_recovery(
    fit: &ExperimentalRecoveryFit,
    spec: &PosteriorSpec,
) -> Result<RecoveredEstimate, ExperimentalFitError> {
    if spec.draws == 0 || spec.burn_in.checked_add(spec.draws).is_none() {
        return Err(ExperimentalFitError::DrawCount);
    }
    let n = fit.nodes.len();
    validate_prior(&spec.prior, n).map_err(ExperimentalFitError::Prior)?;
    let mut row_draws: Vec<Option<Vec<RowDraw>>> = Vec::with_capacity(fit.systems.len());
    let mut identified = Vec::new();
    for (chain, system) in fit.systems.iter().enumerate() {
        let target = fit.nodes[system.target];
        let solution = solve_block(system, &fit.regimes, ConstraintWeighting::Efficient)
            .ok_or(ExperimentalFitError::BlockSolve { target })?;
        match system.rank {
            BlockRank::Full => {
                row_draws.push(Some(posterior_row(fit, system, &solution, spec, chain)?));
            }
            BlockRank::Deficient { .. } => {
                row_draws.push(None);
                if let Some(block) = identified_posterior(fit, system, &solution, &spec.prior)? {
                    identified.push(block);
                }
            }
        }
    }
    let weight = 1.0 / spec.draws as f64;
    let mut inputs = Vec::with_capacity(spec.draws);
    let mut statuses = Vec::with_capacity(spec.draws);
    for draw in 0..spec.draws {
        let mut intercept = vec![f64::NAN; n];
        let mut variance = vec![None; n];
        let blocks: Vec<AssembledCoefficientRecovery> = fit
            .blocks
            .iter()
            .zip(fit.systems.iter().zip(&row_draws))
            .map(|(block, (system, draws))| match (block, draws) {
                (
                    AssembledCoefficientRecovery::Coefficients { target, sources, .. },
                    Some(draws),
                ) => {
                    let row = &draws[draw];
                    intercept[system.target] = row.intercept;
                    variance[system.target] = Some(row.variance);
                    AssembledCoefficientRecovery::Coefficients {
                        target: *target,
                        sources: sources.clone(),
                        values: row.coefficients.clone(),
                    }
                }
                (other, _) => other.clone(),
            })
            .collect();
        let input = draw_input(
            &fit.nodes,
            &blocks,
            &fit.coverage_unresolved,
            intercept,
            diagonal(&variance),
            weight,
        )
        .map_err(ExperimentalFitError::Uncertainty)?;
        statuses.push(fit.status_of(&input));
        inputs.push(input);
    }
    let law = RecoveredCoefficientUncertainty::retain(&fit.nodes, inputs)
        .map_err(ExperimentalFitError::Uncertainty)?;
    Ok(RecoveredEstimate {
        law,
        origin: RecoveredDrawOrigin::Posterior {
            prior: spec.prior.clone(),
            draws: spec.draws,
            burn_in: spec.burn_in,
            seed: spec.seed,
        },
        statuses,
        identified_posteriors: identified,
    })
}

fn validate_prior(prior: &RecoveryPrior, n: usize) -> Result<(), PriorFault> {
    if let CoefficientPrior::Normal { mean, variance } = prior.coefficients
        && !(mean.is_finite() && variance.is_finite() && variance > 0.0)
    {
        return Err(PriorFault::CoefficientPrior);
    }
    match &prior.disturbance {
        DisturbancePrior::Known { variances } => {
            if variances.len() != n {
                return Err(PriorFault::DisturbanceLength { expected: n, found: variances.len() });
            }
            if let Some(node) = variances.iter().position(|v| !(v.is_finite() && *v > 0.0)) {
                return Err(PriorFault::KnownVariance { node });
            }
        }
        DisturbancePrior::InverseWishart { degrees_of_freedom, scale } => {
            if !(degrees_of_freedom.is_finite() && *degrees_of_freedom > 0.0) {
                return Err(PriorFault::DegreesOfFreedom);
            }
            if scale.len() != n {
                return Err(PriorFault::DisturbanceLength { expected: n, found: scale.len() });
            }
            if let Some(node) = scale.iter().position(|v| !(v.is_finite() && *v > 0.0)) {
                return Err(PriorFault::Scale { node });
            }
        }
    }
    Ok(())
}

/// Prior precision (scalar times identity) and prior shift `Λ₀ m₀` on a subspace.
fn prior_terms(prior: CoefficientPrior, basis: Option<&[Vec<f64>]>, dim: usize) -> (f64, Vec<f64>) {
    match prior {
        CoefficientPrior::Flat => (0.0, vec![0.0; dim]),
        CoefficientPrior::Normal { mean, variance } => {
            let precision = 1.0 / variance;
            let shift = match basis {
                None => vec![mean * precision; dim],
                Some(basis) => basis
                    .iter()
                    .map(|vector| vector.iter().sum::<f64>() * mean * precision)
                    .collect(),
            };
            (precision, shift)
        }
    }
}

/// Precision `B/σ² + λ₀ I` and right-hand side `B b̂/σ² + shift`, factored.
fn conditional_gaussian(
    b: &[f64],
    estimate: &[f64],
    precision: f64,
    shift: &[f64],
    variance: f64,
    dim: usize,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut matrix = vec![0.0; dim * dim];
    let mut rhs = shift.to_vec();
    for row in 0..dim {
        for col in 0..dim {
            matrix[row * dim + col] = b[row * dim + col] / variance;
        }
        matrix[row * dim + row] += precision;
        rhs[row] += (0..dim).map(|col| b[row * dim + col] * estimate[col]).sum::<f64>() / variance;
    }
    let mut factor = vec![0.0; dim * dim];
    cholesky_spd_into(&matrix, dim, &mut factor).ok()?;
    let mut scratch = vec![0.0; dim];
    let mut mean = vec![0.0; dim];
    chol_solve_into(&factor, dim, &rhs, &mut scratch, &mut mean);
    Some((factor, mean))
}

fn gaussian_draw(factor: &[f64], mean: &[f64], dim: usize, rng: &mut CausalRng) -> Vec<f64> {
    let z: Vec<f64> = (0..dim).map(|_| standard_normal(rng)).collect();
    let mut shift = vec![0.0; dim];
    back_solve_into(factor, dim, &z, &mut shift);
    mean.iter().zip(&shift).map(|(m, s)| m + s).collect()
}

fn posterior_row(
    fit: &ExperimentalRecoveryFit,
    system: &BlockSystem,
    solution: &BlockSolution,
    spec: &PosteriorSpec,
    chain: usize,
) -> Result<Vec<RowDraw>, ExperimentalFitError> {
    let target = fit.nodes[system.target];
    let refused = ExperimentalFitError::PosteriorPrecision { target };
    let moments =
        fit.residuals[system.target].as_ref().ok_or(ExperimentalFitError::BlockSolve { target })?;
    let k = solution.dim;
    let gram = gram_of_upper(&solution.r, k);
    let (precision, shift) = prior_terms(spec.prior.coefficients, None, k);
    let count = moments.rows as f64;
    let row_of = |values: &[f64]| {
        let mut row = vec![0.0; fit.nodes.len()];
        for (&source, &value) in system.sources.iter().zip(values) {
            row[source] = value;
        }
        residual_vector(system.target, &row)
    };
    let draw_intercept = |values: &[f64], variance: f64, rng: &mut CausalRng| {
        moments.mean_of(&row_of(values)) + (variance / count).sqrt() * standard_normal(rng)
    };
    let mut draws = Vec::with_capacity(spec.draws);
    match &spec.prior.disturbance {
        DisturbancePrior::Known { variances } => {
            let variance = variances[system.target];
            let (factor, mean) =
                conditional_gaussian(&gram, &solution.estimate, precision, &shift, variance, k)
                    .ok_or(refused)?;
            for draw in 0..spec.draws {
                let mut rng = keyed_stream(POSTERIOR_SWEEP_OPERATION, chain, draw, spec.seed);
                let coefficients = gaussian_draw(&factor, &mean, k, &mut rng);
                let intercept = draw_intercept(&coefficients, variance, &mut rng);
                draws.push(RowDraw { coefficients, intercept, variance });
            }
        }
        DisturbancePrior::InverseWishart { degrees_of_freedom, scale } => {
            let scatter = count * moments.variance_of(&row_of(&solution.estimate));
            let prior_scale = scale[system.target];
            let shape = 0.5 * (degrees_of_freedom + (count - 1.0) + k as f64);
            let mut variance = (prior_scale + scatter) / (degrees_of_freedom + count - 1.0);
            for sweep in 0..spec.burn_in + spec.draws {
                let mut rng = keyed_stream(POSTERIOR_SWEEP_OPERATION, chain, sweep, spec.seed);
                let (factor, mean) =
                    conditional_gaussian(&gram, &solution.estimate, precision, &shift, variance, k)
                        .ok_or_else(|| refused.clone())?;
                let coefficients = gaussian_draw(&factor, &mean, k, &mut rng);
                let offset: Vec<f64> =
                    coefficients.iter().zip(&solution.estimate).map(|(b, e)| b - e).collect();
                let distance = quadratic_n(&offset, &gram, &offset, k);
                let rate = 0.5 * (prior_scale + scatter + distance.max(0.0));
                variance = sample_inv_gamma(shape, rate, &mut rng);
                let intercept = draw_intercept(&coefficients, variance, &mut rng);
                if sweep >= spec.burn_in {
                    draws.push(RowDraw { coefficients, intercept, variance });
                }
            }
        }
    }
    Ok(draws)
}

fn identified_posterior(
    fit: &ExperimentalRecoveryFit,
    system: &BlockSystem,
    solution: &BlockSolution,
    prior: &RecoveryPrior,
) -> Result<Option<IdentifiedPosteriorBlock>, ExperimentalFitError> {
    let DisturbancePrior::Known { variances } = &prior.disturbance else {
        return Ok(None);
    };
    let target = fit.nodes[system.target];
    let sources: Vec<VariableId> = system.sources.iter().map(|&s| fit.nodes[s]).collect();
    let basis = solution.basis.clone().unwrap_or_default();
    let d = solution.dim;
    if d == 0 {
        return Ok(Some(IdentifiedPosteriorBlock {
            target,
            sources,
            basis,
            mean: Vec::new(),
            covariance: Vec::new(),
        }));
    }
    let gram = gram_of_upper(&solution.r, d);
    let (precision, shift) = prior_terms(prior.coefficients, Some(basis.as_slice()), d);
    let (factor, mean) = conditional_gaussian(
        &gram,
        &solution.estimate,
        precision,
        &shift,
        variances[system.target],
        d,
    )
    .ok_or(ExperimentalFitError::PosteriorPrecision { target })?;
    Ok(Some(IdentifiedPosteriorBlock {
        target,
        sources,
        basis,
        mean,
        covariance: chol_inverse(&factor, d),
    }))
}
