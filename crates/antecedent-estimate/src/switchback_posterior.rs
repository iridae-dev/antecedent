//! Bayesian counterpart of the switchback treatment effect, under a declared carryover and a
//! declared within-sequence dependence model.
//!
//! # Target
//!
//! [`crate::switchback::switchback_itt`] is a sequence-level Horvitz-Thompson contrast: the
//! average effect over the observed unit-period schedule, with sequences as the independent
//! units and arbitrary within-sequence dependence absorbed by the sandwich. It identifies that
//! effect only under no carryover and no between-sequence interference, and its interval is a
//! design-based statement, not a posterior. A posterior needs a likelihood for the
//! within-sequence dependence, so this module states one and refuses to run without it:
//!
//! * The outcome of sequence `s` is `y_s = X_s b + e_s`, `e_s ~ N(0, sigma2 R_s)` with
//!   independent sequences. `R_s` is the declared correlation: independent, compound
//!   symmetric with a stated intraclass correlation, or first-order autoregressive with a
//!   stated lag-one correlation ([`SequenceDependence`]). The correlation is *declared*, never
//!   estimated; a posterior that also marginalized over it needs a prior on it and is the next
//!   attempt (run the declared values on a grid for a sensitivity reading).
//! * The columns of `X_s` are an intercept, the period's treatment indicator, the previous
//!   period's treatment indicator when first-order carryover is declared (the first period of a
//!   sequence starts from the control state), and optionally fixed period effects for sequences
//!   of equal length ([`PeriodAdjustment`]).
//! * The prior is the diagonal Gaussian and Inverse-Gamma prior of
//!   [`crate::gaussian_linear_posterior`].
//!
//! The estimand is the **constant direct effect `b_A`** of the model, together with the
//! carryover coefficient, their sum (the sustained effect, always-treated against
//! always-control once the carryover has been absorbed) and the period contrasts. When the unit-
//! period effects are heterogeneous, `b_A` is the generalized-least-squares weighted average of
//! them, which is the HT schedule average only for a balanced schedule; the two answer the same
//! target when the effect is constant and differ in general. Unrestricted carryover has no
//! such likelihood (see [`CarryoverDeclaration::Unrestricted`]).
//!
//! # Conditioning
//!
//! The posterior is conditional on the observed assignment. Sequences are randomized
//! independently of outcomes, so the assignment factors out of the likelihood and the posterior
//! does not depend on the randomization probabilities (the likelihood principle). Rows of a
//! sequence must be supplied in period order.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::reason_code;
use antecedent_stats::invert_square;

use crate::error::EstimationError;
use crate::gaussian_linear_posterior::{
    GaussianLinearPrior, PosteriorSummary, StudentPosterior, conjugate_posterior,
};
use crate::util::refuse;

/// Most periods one sequence may have.
pub const MAX_SEQUENCE_PERIODS: usize = 512;

/// Estimand label for evidence records.
pub const SWITCHBACK_POSTERIOR_ESTIMAND: &str =
    "constant_direct_effect_under_declared_carryover_and_dependence";

/// Declared treatment carryover between periods.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CarryoverDeclaration {
    /// A period's outcome depends on that period's treatment only.
    None,
    /// A period's outcome also depends on the previous period's treatment through one added
    /// coefficient; the first period of a sequence starts from the control state.
    FirstOrder,
    /// Carryover of unrestricted order and form. Refused: without a parametric form for the
    /// dependence on the treatment history the direct effect is not identified by a likelihood.
    Unrestricted,
}

/// Declared within-sequence correlation of the errors.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum SequenceDependence {
    /// Independent periods.
    Independent,
    /// Every pair of periods has correlation `rho`, in `(-1 / (m - 1), 1)` for sequences of
    /// `m` periods.
    CompoundSymmetric {
        /// The intraclass correlation.
        rho: f64,
    },
    /// Correlation `rho^|i - j|`, `|rho| < 1`.
    Autoregressive {
        /// The lag-one correlation.
        rho: f64,
    },
    /// No correlation model stated. Refused.
    Unspecified,
}

/// Whether period effects enter the model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PeriodAdjustment {
    /// No period effects.
    None,
    /// A fixed effect per period position (requires sequences of equal length); the period
    /// contrasts against the first period are reported.
    FixedEffects,
}

/// Declared prior of the switchback posterior.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SwitchbackPrior {
    /// Prior precision (relative to `sigma2`) of the treatment and carryover coefficients.
    pub effect_precision: f64,
    /// Prior precision of the intercept and period effects.
    pub nuisance_precision: f64,
    /// Inverse-Gamma shape of `sigma2`.
    pub variance_shape: f64,
    /// Inverse-Gamma scale of `sigma2`.
    pub variance_scale: f64,
}

/// The declared model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SwitchbackModel {
    /// Carryover.
    pub carryover: CarryoverDeclaration,
    /// Within-sequence dependence.
    pub dependence: SequenceDependence,
    /// Period effects.
    pub periods: PeriodAdjustment,
    /// Prior.
    pub prior: SwitchbackPrior,
}

/// Rows of a switchback experiment.
#[derive(Clone, Copy, Debug)]
pub struct SwitchbackRows<'a> {
    /// Outcome of each unit-period.
    pub outcomes: &'a [f64],
    /// Treatment indicator of each unit-period.
    pub assignment: &'a [bool],
    /// Sequence each unit-period belongs to; rows of a sequence are in period order.
    pub sequence_ids: &'a [&'a str],
}

/// What the declarations contributed, kept for evidence synthesis.
#[derive(Clone, Debug, PartialEq)]
pub struct SwitchbackEvidence {
    /// Independent sequences.
    pub sequences: usize,
    /// Unit-period rows.
    pub rows: usize,
    /// The declared carryover.
    pub carryover: CarryoverDeclaration,
    /// The declared dependence.
    pub dependence: SequenceDependence,
    /// Estimand label.
    pub estimand: &'static str,
}

/// Posterior of the switchback effects.
#[derive(Clone, Debug, PartialEq)]
pub struct SwitchbackPosterior {
    /// The direct (current-period) treatment effect.
    pub direct_effect: PosteriorSummary,
    /// The previous-period treatment coefficient, under first-order carryover.
    pub carryover_effect: Option<PosteriorSummary>,
    /// Direct plus carryover, under first-order carryover.
    pub sustained_effect: Option<PosteriorSummary>,
    /// Period `j` against period 0, for `j = 1..`, under fixed period effects.
    pub period_contrasts: Vec<PosteriorSummary>,
    /// Posterior Inverse-Gamma shape and scale of `sigma2`.
    pub variance_shape_and_scale: (f64, f64),
    /// Credible level of every interval.
    pub credible_level: f64,
    /// Declarations and sizes.
    pub evidence: SwitchbackEvidence,
    /// The Student-t marginal behind each summary, for quantile-grid draws.
    pub marginals: SwitchbackMarginals,
}

/// The Student-t marginal posterior of each switchback effect.
#[derive(Clone, Debug, PartialEq)]
pub struct SwitchbackMarginals {
    /// The direct effect.
    pub direct_effect: StudentPosterior,
    /// The carryover coefficient, under first-order carryover.
    pub carryover_effect: Option<StudentPosterior>,
    /// Direct plus carryover, under first-order carryover.
    pub sustained_effect: Option<StudentPosterior>,
    /// Period `j` against period 0, under fixed period effects.
    pub period_contrasts: Vec<StudentPosterior>,
}

fn invalid(detail: &str, message: &str) -> EstimationError {
    refuse(reason_code!("invalid_argument"), detail, message)
}

fn correlation(
    dependence: SequenceDependence,
    periods: usize,
) -> Result<Vec<f64>, EstimationError> {
    let bad = || {
        invalid(
            "switchback_posterior.invalid_correlation",
            "the declared correlation is outside its admissible range for the sequence length",
        )
    };
    let entry: Box<dyn Fn(usize, usize) -> f64> = match dependence {
        SequenceDependence::Independent => {
            Box::new(|i: usize, j: usize| f64::from(u8::from(i == j)))
        }
        SequenceDependence::CompoundSymmetric { rho } => {
            let lower = if periods > 1 { -1.0 / (periods as f64 - 1.0) } else { -1.0 };
            if !(rho.is_finite() && rho > lower && rho < 1.0) {
                return Err(bad());
            }
            Box::new(move |i: usize, j: usize| if i == j { 1.0 } else { rho })
        }
        SequenceDependence::Autoregressive { rho } => {
            if !(rho.is_finite() && rho.abs() < 1.0) {
                return Err(bad());
            }
            Box::new(move |i: usize, j: usize| {
                rho.powi(i32::try_from(i.abs_diff(j)).unwrap_or(i32::MAX))
            })
        }
        SequenceDependence::Unspecified => {
            return Err(refuse(
                reason_code!("cell_not_licensed"),
                "switchback_posterior.dependence_undeclared",
                "a posterior over switchback effects needs a stated within-sequence dependence \
                 model; the design-based sandwich of the frequentist route has no likelihood. \
                 Next attempt: declare independent, compound-symmetric or autoregressive \
                 periods and read the sensitivity over a grid of correlations",
            ));
        }
    };
    let mut matrix = Vec::with_capacity(periods * periods);
    for i in 0..periods {
        for j in 0..periods {
            matrix.push(entry(i, j));
        }
    }
    Ok(matrix)
}

struct Sufficient {
    gram: Vec<f64>,
    xty: Vec<f64>,
    yty: f64,
}

/// Design row of period `t` of a sequence: intercept, treatment, optional lag, optional period
/// dummies.
fn design_row(assignment: &[bool], t: usize, model: &SwitchbackModel, periods: usize) -> Vec<f64> {
    let flag = |b: bool| f64::from(u8::from(b));
    let mut row = vec![1.0, flag(assignment[t])];
    if model.carryover == CarryoverDeclaration::FirstOrder {
        row.push(if t == 0 { 0.0 } else { flag(assignment[t - 1]) });
    }
    if model.periods == PeriodAdjustment::FixedEffects {
        row.extend((1..periods).map(|j| flag(j == t)));
    }
    row
}

fn accumulate(
    sequences: &BTreeMap<&str, Vec<usize>>,
    rows: &SwitchbackRows<'_>,
    model: &SwitchbackModel,
    width: usize,
) -> Result<Sufficient, EstimationError> {
    let mut inverses: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    let mut sufficient =
        Sufficient { gram: vec![0.0; width * width], xty: vec![0.0; width], yty: 0.0 };
    for indices in sequences.values() {
        let m = indices.len();
        let inverse = match inverses.entry(m) {
            std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::btree_map::Entry::Vacant(entry) => {
                let inverse =
                    invert_square(&correlation(model.dependence, m)?, m).ok_or_else(|| {
                        invalid(
                            "switchback_posterior.invalid_correlation",
                            "the declared correlation matrix is singular",
                        )
                    })?;
                entry.insert(inverse)
            }
        };
        let assignment: Vec<bool> = indices.iter().map(|i| rows.assignment[*i]).collect();
        let y: Vec<f64> = indices.iter().map(|i| rows.outcomes[*i]).collect();
        let x: Vec<Vec<f64>> = (0..m).map(|t| design_row(&assignment, t, model, m)).collect();
        // R^-1 applied to the columns of X and to y.
        let apply = |v: &dyn Fn(usize) -> f64| -> Vec<f64> {
            (0..m).map(|i| (0..m).map(|j| inverse[i * m + j] * v(j)).sum()).collect()
        };
        let ri_y = apply(&|j: usize| y[j]);
        sufficient.yty += y.iter().zip(&ri_y).map(|(a, b)| a * b).sum::<f64>();
        for a in 0..width {
            let ri_xa = apply(&|j: usize| x[j][a]);
            sufficient.xty[a] += x.iter().zip(&ri_y).map(|(row, v)| row[a] * v).sum::<f64>();
            for b in 0..width {
                sufficient.gram[a * width + b] +=
                    x.iter().zip(&ri_xa).map(|(row, v)| row[b] * v).sum::<f64>();
            }
        }
    }
    Ok(sufficient)
}

fn group_sequences<'a>(
    rows: &SwitchbackRows<'a>,
    model: &SwitchbackModel,
    credible_level: f64,
) -> Result<BTreeMap<&'a str, Vec<usize>>, EstimationError> {
    if model.carryover == CarryoverDeclaration::Unrestricted {
        return Err(refuse(
            reason_code!("cell_not_licensed"),
            "switchback_posterior.carryover_unrestricted",
            "unrestricted treatment carryover has no likelihood in which the direct effect is \
             identified. Next attempt: declare first-order carryover, or randomize with washout \
             periods that make the no-carryover declaration true",
        ));
    }
    if !(credible_level.is_finite() && credible_level > 0.0 && credible_level < 1.0) {
        return Err(invalid(
            "switchback_posterior.invalid_level",
            "the credible level must lie strictly between zero and one",
        ));
    }
    let n = rows.outcomes.len();
    if n == 0 || rows.assignment.len() != n || rows.sequence_ids.len() != n {
        return Err(invalid(
            "switchback_posterior.shape",
            "outcomes, assignment and sequence ids must have equal non-zero length",
        ));
    }
    if rows.outcomes.iter().any(|y| !y.is_finite())
        || rows.sequence_ids.iter().any(|s| s.is_empty())
    {
        return Err(invalid(
            "switchback_posterior.invalid_outcome",
            "outcomes must be finite and sequence ids non-empty",
        ));
    }
    let treated = rows.assignment.iter().filter(|a| **a).count();
    if treated == 0 || treated == n {
        return Err(invalid(
            "switchback_posterior.single_arm",
            "both treatment states must be observed",
        ));
    }
    let mut sequences: BTreeMap<&'a str, Vec<usize>> = BTreeMap::new();
    for (index, id) in rows.sequence_ids.iter().enumerate() {
        sequences.entry(*id).or_default().push(index);
    }
    let longest = sequences.values().map(Vec::len).max().unwrap_or(0);
    let shortest = sequences.values().map(Vec::len).min().unwrap_or(0);
    if sequences.len() < 2 {
        return Err(invalid(
            "switchback_posterior.too_few_sequences",
            "at least two independent sequences are required",
        ));
    }
    if longest > MAX_SEQUENCE_PERIODS {
        return Err(invalid(
            "switchback_posterior.too_many_periods",
            "a sequence has more periods than the supported maximum",
        ));
    }
    if model.periods == PeriodAdjustment::FixedEffects && longest != shortest {
        return Err(invalid(
            "switchback_posterior.unbalanced_periods",
            "period effects need sequences of equal length",
        ));
    }
    Ok(sequences)
}

/// Posterior of the switchback effects under the declared model.
///
/// # Errors
///
/// `cell_not_licensed` with `switchback_posterior.carryover_unrestricted` for unrestricted
/// carryover and `switchback_posterior.dependence_undeclared` for an unspecified dependence (each
/// names the next attempt); `invalid_argument` with `switchback_posterior.shape`,
/// `.invalid_outcome`, `.single_arm`, `.too_few_sequences`, `.unbalanced_periods`,
/// `.too_many_periods`, `.invalid_correlation` or `.invalid_level`, and with the
/// `gaussian_linear_posterior.*` details for an invalid prior.
pub fn posterior_switchback(
    rows: &SwitchbackRows<'_>,
    model: &SwitchbackModel,
    credible_level: f64,
) -> Result<SwitchbackPosterior, EstimationError> {
    let sequences = group_sequences(rows, model, credible_level)?;
    let n = rows.outcomes.len();
    let longest = sequences.values().map(Vec::len).max().unwrap_or(0);
    let lag = model.carryover == CarryoverDeclaration::FirstOrder;
    let period_columns =
        if model.periods == PeriodAdjustment::FixedEffects { longest - 1 } else { 0 };
    let width = 2 + usize::from(lag) + period_columns;
    let sufficient = accumulate(&sequences, rows, model, width)?;
    let prior = &model.prior;
    let precision = (0..width)
        .map(|j| {
            if j == 1 || (lag && j == 2) {
                prior.effect_precision
            } else {
                prior.nuisance_precision
            }
        })
        .collect();
    let posterior = conjugate_posterior(
        &sufficient.gram,
        &sufficient.xty,
        sufficient.yty,
        n,
        &GaussianLinearPrior {
            coefficient_precision: precision,
            variance_shape: prior.variance_shape,
            variance_scale: prior.variance_scale,
        },
    )?;
    let marginal = |weights: Vec<f64>| {
        posterior.linear_functional(&weights).ok_or_else(|| {
            invalid(
                "switchback_posterior.degenerate_contrast",
                "the contrast has zero posterior scale",
            )
        })
    };
    let unit = |positions: &[usize]| {
        let mut weights = vec![0.0; width];
        for p in positions {
            weights[*p] = 1.0;
        }
        weights
    };
    let direct = marginal(unit(&[1]))?;
    let (carryover, sustained) = if lag {
        (Some(marginal(unit(&[2]))?), Some(marginal(unit(&[1, 2]))?))
    } else {
        (None, None)
    };
    let first_period = 2 + usize::from(lag);
    let periods = (0..period_columns)
        .map(|j| marginal(unit(&[first_period + j])))
        .collect::<Result<Vec<_>, _>>()?;
    let level = credible_level;
    Ok(SwitchbackPosterior {
        direct_effect: direct.summary(level),
        carryover_effect: carryover.map(|t| t.summary(level)),
        sustained_effect: sustained.map(|t| t.summary(level)),
        period_contrasts: periods.iter().map(|t| t.summary(level)).collect(),
        variance_shape_and_scale: posterior.variance_shape_and_scale(),
        credible_level,
        evidence: SwitchbackEvidence {
            sequences: sequences.len(),
            rows: n,
            carryover: model.carryover,
            dependence: model.dependence,
            estimand: SWITCHBACK_POSTERIOR_ESTIMAND,
        },
        marginals: SwitchbackMarginals {
            direct_effect: direct,
            carryover_effect: carryover,
            sustained_effect: sustained,
            period_contrasts: periods,
        },
    })
}
