// SPDX-License-Identifier: MIT OR Apache-2.0
//! Finite-candidate likelihood fitting for polynomial equilibrium systems.
//! Each candidate declares a domain and a root-selection law through stable
//! geometric regions. Independent Gaussian measurement noise defines an actual
//! likelihood. Root enclosures yield likelihood bounds; unresolved roots or
//! overlapping likelihood bounds withhold an optimum certificate.

use antecedent_core::ExecutionContext;
use antecedent_model::{
    equation_system::{Interval, IsolatedSolutions, IsolationPolicy, PolynomialSystem},
    nonlinear_equilibrium::EquilibriumError,
};
use serde::{Deserialize, Serialize};

/// A scientifically specified equilibrium-selection region and its mass.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RootSelectionRegion {
    /// One finite region bound per variable; must contain exactly one certified root box.
    pub bounds: Vec<Interval>,
    /// Explicit mass of that equilibrium, including zero for excluded branches.
    pub probability: f64,
}

/// One fully specified parameter candidate with an explicit equilibrium law.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolynomialCandidate {
    /// Stable parameter/scenario identity, not a solver initialization.
    pub id: String,
    /// Named equations at this parameter point and bounded domain.
    pub model: PolynomialSystem,
    /// Selection mass tied to geometric root regions, not solver root ordering.
    pub selection: Vec<RootSelectionRegion>,
}

/// Measurement of an independently selected equilibrium.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolynomialMeasurement {
    /// Measured endogenous state.
    pub values: Vec<f64>,
    /// Known independent positive Gaussian measurement variances.
    pub variances: Vec<f64>,
}

/// Candidate-specific likelihood enclosure and all solver outcomes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolynomialLikelihoodBound {
    /// Parameter candidate identity.
    pub candidate: String,
    /// Lower log-likelihood bound, absent while root isolation is unresolved.
    pub lower: Option<f64>,
    /// Upper log-likelihood bound, absent while root isolation is unresolved.
    pub upper: Option<f64>,
    /// Outward-rounded unnormalized log-likelihood score when selection has one
    /// probability-one branch; Gaussian normalization cancels across candidates.
    /// Mixture-transcendental evaluation has no rigorous ranking certificate here.
    pub ranking_score: Option<Interval>,
    /// Certified and unresolved root regions, retained rather than dropped.
    pub solutions: IsolatedSolutions,
}

/// Complete bounded-candidate likelihood selection attempt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolynomialFit {
    /// Best lower-bound candidate, absent if no candidate law was fully evaluated.
    pub best_candidate: Option<String>,
    /// True only when the candidate lower bound exceeds every rival upper bound.
    pub finite_set_optimum_certified: bool,
    /// All evaluated/unevaluated candidate likelihood enclosures.
    pub candidates: Vec<PolynomialLikelihoodBound>,
    /// Following numerical attempt when ranking could not be certified.
    pub next_attempt: Option<String>,
}
fn invalid(s: &str) -> EquilibriumError {
    EquilibriumError::Invalid(s.into())
}
fn logsumexp(values: &[f64]) -> f64 {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    max + values.iter().map(|v| (v - max).exp()).sum::<f64>().ln()
}

/// Evaluate and rank an explicit finite parameter set. No continuous/global
/// optimum claim is made. Measurements do not estimate a selection law: each
/// candidate supplies that law before it is scored.
#[allow(
    clippy::too_many_lines,
    reason = "root certification, geometric selection binding and finite-candidate likelihood comparison are one bounded procedure"
)]
#[allow(
    clippy::float_cmp,
    reason = "a rigorous deterministic-branch ranking requires probability exactly one, not approximately one"
)]
pub fn fit_polynomial_candidates(
    candidates: &[PolynomialCandidate],
    measurements: &[PolynomialMeasurement],
    policy: IsolationPolicy,
    ctx: &ExecutionContext,
) -> Result<PolynomialFit, EquilibriumError> {
    if candidates.is_empty()
        || candidates.len() > 256
        || measurements.is_empty()
        || measurements.len() > 10_000
    {
        return Err(invalid("constraint candidate/data limits"));
    }
    let n = candidates[0].model.variables.len();
    let mut ids = std::collections::BTreeSet::new();
    for candidate in candidates {
        candidate.model.validate()?;
        if candidate.model.variables != candidates[0].model.variables
            || candidate.id.is_empty()
            || candidate.id.len() > 256
            || !ids.insert(&candidate.id)
            || candidate.selection.is_empty()
            || candidate.selection.len() > 1024
        {
            return Err(invalid(
                "constraint candidates need common coordinates and unique identities",
            ));
        }
        if candidate.selection.iter().any(|s| {
            s.bounds.len() != n
                || s.bounds
                    .iter()
                    .any(|b| !b.lower.is_finite() || !b.upper.is_finite() || b.lower > b.upper)
                || !s.probability.is_finite()
                || s.probability < 0.0
        }) || (candidate.selection.iter().map(|s| s.probability).sum::<f64>() - 1.0).abs()
            > 1e-12
        {
            return Err(invalid("explicit normalized root-region selection law required"));
        }
    }
    if measurements.iter().any(|r| {
        r.values.len() != n
            || r.variances.len() != n
            || r.values.iter().any(|v| !v.is_finite())
            || r.variances.iter().any(|v| !v.is_finite() || *v <= 0.0)
    }) {
        return Err(invalid("constraint Gaussian observation model"));
    }
    let mut results = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if ctx.cancellation.is_cancelled() {
            return Err(EquilibriumError::Resource("cancelled".into()));
        }
        let solutions = candidate.model.isolate(policy, ctx)?;
        if !solutions.unresolved.is_empty() || solutions.roots.is_empty() {
            results.push(PolynomialLikelihoodBound {
                candidate: candidate.id.clone(),
                lower: None,
                upper: None,
                ranking_score: None,
                solutions,
            });
            continue;
        }
        let mut root_probabilities = Vec::new();
        let mut selected = std::collections::BTreeSet::new();
        for root in &solutions.roots {
            let matches: Vec<_> = candidate
                .selection
                .iter()
                .enumerate()
                .filter(|(_, s)| {
                    s.bounds
                        .iter()
                        .zip(root)
                        .all(|(region, r)| region.lower <= r.lower && region.upper >= r.upper)
                })
                .collect();
            if matches.len() != 1 || !selected.insert(matches[0].0) {
                return Err(invalid(
                    "selection regions must identify roots uniquely without overlap",
                ));
            }
            root_probabilities.push(matches[0].1.probability);
        }
        if selected.len() != candidate.selection.len() {
            return Err(invalid("a declared selection region contains no certified root"));
        }
        let selected_root = if root_probabilities.iter().filter(|&&p| p > 0.0).count() == 1 {
            root_probabilities.iter().position(|&p| p == 1.0)
        } else {
            None
        };
        let mut ranking_score = selected_root.map(|_| Interval { lower: 0.0, upper: 0.0 });
        if let (Some(index), Some(score)) = (selected_root, ranking_score.as_mut()) {
            for observed in measurements {
                for ((&root, &value), &variance) in
                    solutions.roots[index].iter().zip(&observed.values).zip(&observed.variances)
                {
                    *score = score.add_outward(
                        root.standardized_squared_distance(value, variance)?.scale_outward(-0.5),
                    );
                }
            }
            if !score.lower.is_finite() || !score.upper.is_finite() {
                return Err(invalid("constraint score exceeds finite numerical range"));
            }
        }
        let mut lower = 0.0;
        let mut upper = 0.0;
        for observed in measurements {
            let mut low_terms = Vec::new();
            let mut high_terms = Vec::new();
            for (root, &probability) in solutions.roots.iter().zip(&root_probabilities) {
                if probability == 0.0 {
                    continue;
                }
                let mut low = probability.ln();
                let mut high = low;
                for ((r, &y), &variance) in
                    root.iter().zip(&observed.values).zip(&observed.variances)
                {
                    let minimum = if y < r.lower {
                        r.lower - y
                    } else if y > r.upper {
                        y - r.upper
                    } else {
                        0.0
                    };
                    let maximum = (y - r.lower).abs().max((y - r.upper).abs());
                    let normalizer = (2.0 * std::f64::consts::PI * variance).ln();
                    low -= 0.5 * (maximum * maximum / variance + normalizer);
                    high -= 0.5 * (minimum * minimum / variance + normalizer);
                }
                low_terms.push(low);
                high_terms.push(high);
            }
            lower += logsumexp(&low_terms);
            upper += logsumexp(&high_terms);
        }
        if !lower.is_finite() || !upper.is_finite() {
            return Err(invalid("constraint likelihood exceeds finite numerical range"));
        }
        // Allow a conservative floating evaluation margin in addition to root uncertainty.
        let margin = 1e-10 * (1.0 + lower.abs().max(upper.abs()));
        results.push(PolynomialLikelihoodBound {
            candidate: candidate.id.clone(),
            lower: Some(lower - margin),
            upper: Some(upper + margin),
            ranking_score,
            solutions,
        });
    }
    let best = results
        .iter()
        .enumerate()
        .filter_map(|(i, r)| r.lower.map(|l| (i, l)))
        .max_by(|a, b| a.1.total_cmp(&b.1));
    let certified = best.is_some_and(|(i, _)| {
        results[i].ranking_score.is_some_and(|score| {
            results.iter().enumerate().all(|(j, r)| {
                i == j || r.ranking_score.is_some_and(|other| score.lower > other.upper)
            })
        })
    });
    Ok(PolynomialFit {best_candidate:best.map(|(i,_)|results[i].candidate.clone()),finite_set_optimum_certified:certified,candidates:results,next_attempt:(!certified).then(||"refine unresolved root domains/enclosures and repeat the same finite-candidate comparison; mixture laws additionally require validated transcendental likelihood enclosures before an optimum certificate".into())})
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_model::equation_system::{Monomial, PolynomialEquation};
    #[test]
    fn finite_parameter_fit_uses_declared_positive_root_selection_and_measurement_likelihood() {
        let candidates: Vec<_> = [0.25, 0.81]
            .into_iter()
            .map(|a| PolynomialCandidate {
                id: format!("a={a}"),
                model: PolynomialSystem {
                    variables: vec!["x".into()],
                    equations: vec![PolynomialEquation {
                        id: "square".into(),
                        terms: vec![
                            Monomial { coefficient: 1.0, powers: vec![2] },
                            Monomial { coefficient: -a, powers: vec![0] },
                        ],
                    }],
                    domain: vec![Interval { lower: -2.0, upper: 2.0 }],
                },
                selection: vec![
                    RootSelectionRegion {
                        bounds: vec![Interval { lower: -2.0, upper: 0.0 }],
                        probability: 0.0,
                    },
                    RootSelectionRegion {
                        bounds: vec![Interval { lower: 0.0, upper: 2.0 }],
                        probability: 1.0,
                    },
                ],
            })
            .collect();
        let data: Vec<_> = [0.88, 0.91, 0.89, 0.92]
            .into_iter()
            .map(|y| PolynomialMeasurement { values: vec![y], variances: vec![0.01] })
            .collect();
        let ctx = ExecutionContext::for_tests(1);
        let fit = fit_polynomial_candidates(&candidates, &data, Default::default(), &ctx).unwrap();
        assert_eq!(fit.best_candidate.as_deref(), Some("a=0.81"));
        assert!(fit.finite_set_optimum_certified);
        let unresolved = fit_polynomial_candidates(
            &candidates,
            &data,
            IsolationPolicy { max_boxes: 1, ..Default::default() },
            &ctx,
        )
        .unwrap();
        assert!(!unresolved.finite_set_optimum_certified);
        assert!(unresolved.next_attempt.is_some());
    }
}
