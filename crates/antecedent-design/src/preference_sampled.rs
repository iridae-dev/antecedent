//! Expected-value margins across a preference set under a measure over structures some of whose
//! equivalence classes were sampled.
//!
//! The value is affine in the trade-off parameters on the exact route, so every pairwise margin
//! is affine too and the vertices of the preference set bound it over the whole set. At each
//! vertex the margin of one action over another is the weight-summed margin over the evaluated
//! structures, with the Monte Carlo standard error of the members drawn from each sampled class.

// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_io_core::distribution_artifact::DistributionArtifact;
use antecedent_stats::linear_program::{self, ConstraintSense, LinearConstraint};

use crate::decision_eval::utility_draws;
use crate::preference_contract::PreferenceContract;
use crate::preference_decision::PreferenceDecisionOptions;
use crate::preference_model::PreferenceSet;

/// The Monte Carlo estimate of the expected-value margin of one action over another at one
/// vertex of the preference set, under the declared measure over structures, with its Monte
/// Carlo standard error. Margins are affine in the trade-off parameters, so the vertices
/// bound them over the whole set.
#[derive(Clone, Debug, PartialEq)]
pub struct SampledMargin {
    /// The earlier action in declaration order.
    pub first: String,
    /// The later action in declaration order.
    pub second: String,
    /// The vertex of the preference set, in parameter order.
    pub vertex: Vec<f64>,
    /// Mass-weighted expected value of `first` minus that of `second` over the evaluated
    /// structures, in the value model's utility units (for a loss criterion smaller values
    /// are better).
    pub margin: f64,
    /// Monte Carlo standard error from the sampled classes; `None` when a sampled class drew
    /// a single member.
    pub standard_error: Option<f64>,
}

/// One structure of the measure: its weight and, when it was evaluated, the expectation of every
/// part of the linear form for every action (`[action][part]`, part zero the fixed part).
#[derive(Clone, Debug, PartialEq)]
pub struct SampledStructure {
    /// Weight of the structure under the declared measure.
    pub weight: f64,
    /// Part expectations; `None` for a structure that was not evaluated, which adds nothing to
    /// a margin.
    pub expectations: Option<Vec<Vec<f64>>>,
}

/// One sampled equivalence class: its mass, the members drawn, and the structures that are its
/// drawn members.
#[derive(Clone, Debug, PartialEq)]
pub struct SampledClass {
    /// Mass of the class under the declared measure.
    pub mass: f64,
    /// Members drawn, with replacement.
    pub drawn: usize,
    /// `(structure index, times drawn)` for each distinct member drawn.
    pub members: Vec<(usize, usize)>,
}

/// The expectation of every part of the contract's linear form for every action on one aligned
/// joint law: `[action][part]`, part zero the fixed part, each the weighted mean of the part's
/// per-draw values. `None` when the model has no linear form or the law does not answer the
/// contract's term contract ([`PreferenceContract::term_contract`]).
#[must_use]
pub fn part_expectations(
    contract: &PreferenceContract,
    law: &DistributionArtifact,
) -> Option<Vec<Vec<f64>>> {
    let terms = contract.term_contract().ok()??;
    let (actions, parts) = (contract.actions.len(), contract.model.preferences.dimension() + 1);
    let draws = utility_draws(&terms, law).ok()?;
    let mean = |column: &[f64]| -> f64 {
        column.iter().zip(&draws.weights).map(|(value, weight)| value * weight).sum()
    };
    (0..actions)
        .map(|action| {
            (0..parts)
                .map(|part| draws.utilities.get(action * parts + part).map(|c| mean(c.as_slice())))
                .collect::<Option<Vec<f64>>>()
        })
        .collect()
}

/// The vertices of a preference set: the declared ranges cut by the linear relations. `None`
/// when they cannot be enumerated within the options' vertex cap.
#[must_use]
pub fn preference_set_vertices(
    set: &PreferenceSet,
    options: PreferenceDecisionOptions,
) -> Option<Vec<Vec<f64>>> {
    if set.dimension() == 0 {
        return Some(vec![Vec::new()]);
    }
    let rows = set.inequality_rows();
    let constraints: Vec<LinearConstraint> = rows
        .coefficients
        .into_iter()
        .zip(rows.rhs)
        .map(|(coefficients, rhs)| LinearConstraint {
            coefficients,
            sense: ConstraintSense::AtMost,
            rhs,
        })
        .collect();
    let (lower, upper): (Vec<f64>, Vec<f64>) = set.bounds().into_iter().unzip();
    linear_program::vertices(&constraints, &lower, &upper, options.vertex_cap, options.lp).ok()
}

/// Every pairwise margin at every vertex of the contract's preference set, with the Monte Carlo
/// standard error of the sampled classes.
///
/// The margin of `first` over `second` at a vertex is `sum_s w_s (c_first(s) - c_second(s))`
/// over the structures, `c(s)` the vertex value of the part expectations; a structure without
/// expectations adds nothing. For each sampled class of mass `m` with `n` draws, the margin
/// values of its drawn members (each counted as often as it was drawn, draws not among the
/// listed members counted as zero) give `m^2 * sum (v - mean)^2 / (n - 1) / n` to the variance;
/// a class with fewer than two draws leaves the standard error undefined. Empty when the
/// vertices cannot be enumerated.
#[must_use]
#[allow(clippy::cast_precision_loss, reason = "draw counts are far below 2^53")]
pub fn sampled_margins(
    contract: &PreferenceContract,
    structures: &[SampledStructure],
    classes: &[SampledClass],
    options: PreferenceDecisionOptions,
) -> Vec<SampledMargin> {
    let Some(vertices) = preference_set_vertices(&contract.model.preferences, options) else {
        return Vec::new();
    };
    let actions = contract.actions.len();
    let margin_at = |index: usize, (first, second): (usize, usize), vertex: &[f64]| -> f64 {
        structures.get(index).and_then(|s| s.expectations.as_ref()).map_or(0.0, |e| {
            let difference = |part: usize| e[first][part] - e[second][part];
            difference(0)
                + vertex.iter().enumerate().map(|(j, theta)| theta * difference(j + 1)).sum::<f64>()
        })
    };
    let mut out = Vec::new();
    for vertex in &vertices {
        for first in 0..actions {
            for second in (first + 1)..actions {
                let margin: f64 = structures
                    .iter()
                    .enumerate()
                    .map(|(index, s)| s.weight * margin_at(index, (first, second), vertex))
                    .sum();
                let mut variance = 0.0;
                let mut defined = true;
                for class in classes {
                    let drawn = class.drawn;
                    if drawn < 2 {
                        defined = false;
                        continue;
                    }
                    let values: Vec<(f64, usize)> = class
                        .members
                        .iter()
                        .map(|&(index, k)| (margin_at(index, (first, second), vertex), k))
                        .collect();
                    let n = drawn as f64;
                    let mean = values.iter().map(|(v, k)| v * *k as f64).sum::<f64>() / n;
                    let squares: f64 =
                        values.iter().map(|(v, k)| (v - mean).powi(2) * *k as f64).sum();
                    // Draws not among the listed members (none, when every draw compiled)
                    // contribute zero.
                    let listed: usize = values.iter().map(|(_, k)| *k).sum();
                    let rest = drawn.saturating_sub(listed) as f64 * mean.powi(2);
                    variance += class.mass * class.mass * (squares + rest) / (n - 1.0) / n;
                }
                out.push(SampledMargin {
                    first: contract.actions[first].id.clone(),
                    second: contract.actions[second].id.clone(),
                    vertex: vertex.clone(),
                    margin,
                    standard_error: defined.then(|| variance.sqrt()),
                });
            }
        }
    }
    out
}
