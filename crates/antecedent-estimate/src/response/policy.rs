//! Discrete-intervention and policy support for g-computation responses.
//!
//! The intervention-law atoms (exact discrete support plus a Gauss–Hermite
//! rule for a Gaussian policy), the additive and exact finite-mixture
//! g-computation row expectations, and the static-Bayesian policy reduction.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{Intervention, StochasticPolicy, VariableId};
use antecedent_stats::gauss_hermite_standard_normal;

use super::{CompleteSample, MAX_EXACT_MIXTURE_COMBINATIONS, predict_one};
use crate::EstimationError;

/// Nodes of the Gauss–Hermite rule applied to a Gaussian intervention policy. The
/// dose-response smooths are cubic B-splines, so the integrand is piecewise cubic;
/// 48 nodes resolve every knot interval a policy spans to well below the SE.
const POLICY_QUADRATURE_NODES: usize = 48;

pub(super) fn intervention_needs_quadrature(intervention: &Intervention) -> bool {
    matches!(
        intervention,
        Intervention::Stochastic { policy: StochasticPolicy::Gaussian { .. }, .. }
    )
}

/// Atoms of an intervention's law: the exact support for discrete policies and a
/// Gauss–Hermite rule for a Gaussian policy.
pub(super) fn policy_support(
    intervention: &Intervention,
) -> Result<Vec<DiscreteAtom>, EstimationError> {
    if let Intervention::Stochastic {
        policy: StochasticPolicy::Gaussian { mean, variance }, ..
    } = intervention
    {
        if !mean.is_finite() || !variance.is_finite() || *variance < 0.0 {
            return Err(EstimationError::unsupported(
                "Gaussian intervention needs a finite mean and a finite non-negative variance",
            ));
        }
        let (nodes, weights) = gauss_hermite_standard_normal(POLICY_QUADRATURE_NODES);
        let sd = variance.sqrt();
        return Ok(nodes
            .into_iter()
            .zip(weights)
            .map(|(node, weight)| DiscreteAtom::Level { value: mean + sd * node, weight })
            .collect());
    }
    discrete_intervention_support(intervention)
}

/// Policy expectation of the additive outcome model for every row.
///
/// With `μ(a, x) = α + Σ_k f_k(a_k) + g(x)`, `E[μ] = Σ_k E[μ(A_k, others factual)]
/// − (k − 1)·μ(factual)`: each policy is integrated over its own support and the
/// joint support is never formed, so cost is `n · Σ_k |support_k|`.
pub(super) fn additive_policy_rows(
    fit: &antecedent_stats::GamFit,
    sample: &CompleteSample,
    interventions: &[Intervention],
) -> Result<Vec<f64>, EstimationError> {
    let supports: Vec<Vec<DiscreteAtom>> =
        interventions.iter().map(policy_support).collect::<Result<_, _>>()?;
    let extra_terms = supports.len().saturating_sub(1) as f64;
    let mut factual = vec![0.0; sample.raw_cols];
    let mut row = vec![0.0; sample.raw_cols];
    let mut out = Vec::with_capacity(sample.len());
    for row_index in 0..sample.len() {
        sample.write_raw_row(row_index, &mut factual);
        row.copy_from_slice(&factual);
        let mut total = -extra_terms * predict_one(fit, &factual)?;
        for (column, support) in supports.iter().enumerate() {
            for atom in support {
                let (level, weight) = match *atom {
                    DiscreteAtom::Level { value, weight } => (value, weight),
                    DiscreteAtom::Shift { delta } => (factual[column] + delta, 1.0),
                };
                row[column] = level;
                total += weight * predict_one(fit, &row)?;
            }
            row[column] = factual[column];
        }
        out.push(total);
    }
    Ok(out)
}

/// One atom of a discrete intervention law: absolute level, or additive shift of the factual.
#[derive(Clone, Copy, Debug)]
pub(super) enum DiscreteAtom {
    /// Absolute treatment level with mixture weight.
    Level { value: f64, weight: f64 },
    /// Additive shift of the unit's factual treatment (weight is always 1).
    Shift { delta: f64 },
}

/// Exact finite-mixture g-computation for Set/Shift/Bernoulli/Categorical policies.
///
/// Bernoulli and Categorical are summed over their support with the declared probabilities
/// rather than Monte-Carlo sampled through a continuous smoother. That avoids treating
/// unordered category codes as ordered coordinates along a spline.
pub(super) fn exact_discrete_intervention_rows(
    fit: &antecedent_stats::GamFit,
    sample: &CompleteSample,
    interventions: &[Intervention],
) -> Result<Vec<f64>, EstimationError> {
    let supports: Vec<Vec<DiscreteAtom>> =
        interventions.iter().map(discrete_intervention_support).collect::<Result<_, _>>()?;
    // The mixture is a cartesian product across interventions, so its cost is exponential in
    // how many discrete policies are joined. The Monte-Carlo path it replaced was bounded at
    // a fixed draw count, so without a budget here a query that used to return in
    // milliseconds can run for hours. Refuse rather than silently reverting to an
    // approximation the caller did not ask for.
    let combinations = supports
        .iter()
        .try_fold(1usize, |product, support| product.checked_mul(support.len()))
        .filter(|product| *product <= MAX_EXACT_MIXTURE_COMBINATIONS);
    if combinations.is_none() {
        return Err(EstimationError::unsupported(
            "joint discrete intervention support exceeds the exact-mixture budget; intervene on fewer variables or coarsen the category supports",
        ));
    }
    let mut out = Vec::with_capacity(sample.len());
    let mut row = vec![0.0; sample.raw_cols];
    for row_index in 0..sample.len() {
        sample.write_raw_row(row_index, &mut row);
        out.push(mixture_expectation(fit, &mut row, &supports, 0, 1.0)?);
    }
    Ok(out)
}

pub(super) fn discrete_intervention_support(
    intervention: &Intervention,
) -> Result<Vec<DiscreteAtom>, EstimationError> {
    let numeric = |value: &antecedent_core::Value| {
        value.as_f64().filter(|number| number.is_finite()).ok_or_else(|| {
            EstimationError::unsupported("intervention response requires finite numeric values")
        })
    };
    match intervention {
        Intervention::Set { value, .. } => {
            Ok(vec![DiscreteAtom::Level { value: numeric(value)?, weight: 1.0 }])
        }
        Intervention::Shift { delta, .. } => {
            Ok(vec![DiscreteAtom::Shift { delta: numeric(delta)? }])
        }
        Intervention::Stochastic { policy: StochasticPolicy::Bernoulli { p }, .. } => {
            if !p.is_finite() || !(0.0..=1.0).contains(p) {
                return Err(EstimationError::unsupported(
                    "Bernoulli intervention probability must lie in [0, 1]",
                ));
            }
            Ok([(0.0, 1.0 - p), (1.0, *p)]
                .into_iter()
                .filter(|(_, w)| *w > 0.0)
                .map(|(value, weight)| DiscreteAtom::Level { value, weight })
                .collect())
        }
        Intervention::Stochastic { policy: StochasticPolicy::Categorical { probs }, .. } => {
            let total: f64 = probs.iter().sum();
            if !total.is_finite()
                || total <= 0.0
                || probs.iter().any(|p| !p.is_finite() || *p < 0.0)
            {
                return Err(EstimationError::unsupported(
                    "Categorical intervention probabilities must be finite and non-negative",
                ));
            }
            Ok(probs
                .iter()
                .enumerate()
                .filter(|(_, p)| **p > 0.0)
                .map(|(index, p)| DiscreteAtom::Level { value: index as f64, weight: p / total })
                .collect())
        }
        _ => Err(EstimationError::unsupported(
            "exact discrete intervention mixture does not cover this policy",
        )),
    }
}

fn mixture_expectation(
    fit: &antecedent_stats::GamFit,
    row: &mut [f64],
    supports: &[Vec<DiscreteAtom>],
    column: usize,
    weight: f64,
) -> Result<f64, EstimationError> {
    if !(weight.is_finite() && weight >= 0.0) {
        return Err(EstimationError::unsupported(
            "intervention mixture weight must be finite and non-negative",
        ));
    }
    if column == supports.len() {
        return Ok(weight * predict_one(fit, row)?);
    }
    let mut sum = 0.0;
    let factual = row[column];
    for atom in &supports[column] {
        let saved = row[column];
        let branch = match *atom {
            DiscreteAtom::Level { value, weight: atom_weight } => {
                row[column] = value;
                atom_weight
            }
            DiscreteAtom::Shift { delta } => {
                row[column] = factual + delta;
                1.0
            }
        };
        sum += mixture_expectation(fit, row, supports, column + 1, weight * branch)?;
        row[column] = saved;
    }
    Ok(sum)
}

/// Expected policy level for a linear-additive response. Integrating each
/// coefficient draw at this level is exact within that model, rather than a
/// deterministic replacement of the policy in a nonlinear response estimator.
pub(super) fn static_bayesian_policy(
    iv: &Intervention,
) -> Result<(VariableId, Option<f64>, f64), EstimationError> {
    let numeric = |value: &antecedent_core::Value| {
        value.as_f64().filter(|x| x.is_finite()).ok_or_else(|| {
            EstimationError::unsupported("static Bayesian policy requires finite numeric values")
        })
    };
    let (target, level, shift) = match iv {
        Intervention::Set { variable, value } => (*variable, Some(numeric(value)?), 0.0),
        Intervention::Shift { variable, delta } => (*variable, None, numeric(delta)?),
        Intervention::Stochastic { variable, policy } => {
            let mean = match policy {
                StochasticPolicy::Bernoulli { p } => *p,
                StochasticPolicy::Gaussian { mean, .. } => *mean,
                StochasticPolicy::Categorical { probs } => {
                    // Scale before summing: valid finite probabilities need not
                    // be normalized and their raw sum can overflow.
                    let scale = probs.iter().copied().fold(0.0_f64, f64::max);
                    let total: f64 = probs.iter().map(|p| p / scale).sum();
                    probs.iter().enumerate().map(|(i, p)| i as f64 * (p / scale) / total).sum()
                }
                _ => {
                    return Err(EstimationError::unsupported(
                        "unsupported static Bayesian stochastic policy",
                    ));
                }
            };
            (*variable, Some(mean), 0.0)
        }
        Intervention::Soft { variable, mechanism } => {
            if mechanism.parameters.len() != 1 || !mechanism.parameters[0].is_finite() {
                return Err(EstimationError::unsupported(
                    "static Bayesian Soft requires one finite parameter",
                ));
            }
            match mechanism.family_id.as_ref() {
                "constant" => (*variable, Some(mechanism.parameters[0]), 0.0),
                "additive_shift" => (*variable, None, mechanism.parameters[0]),
                _ => {
                    return Err(EstimationError::unsupported(
                        "static Bayesian Soft supports constant and additive_shift",
                    ));
                }
            }
        }
        _ => {
            return Err(EstimationError::unsupported(
                "unsupported static Bayesian intervention policy",
            ));
        }
    };
    if level.is_some_and(|x| !x.is_finite()) || !shift.is_finite() {
        return Err(EstimationError::unsupported("static Bayesian policy has a non-finite mean"));
    }
    Ok((target, level, shift))
}
