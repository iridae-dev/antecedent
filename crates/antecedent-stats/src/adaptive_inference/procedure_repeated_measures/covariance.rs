//! Declared within-unit covariance, summary reduction and simulation.

use super::{OutcomeStructure, UnitSummary, WithinUnitCorrelation};
use antecedent_core::CausalRng;
use antecedent_kernels::standard_normal;

// -- Covariance -----------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Dependence {
    CompoundSymmetric(f64),
    Autoregressive(f64),
    RandomEffect { between: f64, within: f64 },
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Covariance {
    pub(super) measurements: usize,
    pub(super) marginal_variance: f64,
    dependence: Dependence,
    /// Row-major `m x m` correlation matrix.
    correlation: Vec<f64>,
    /// Row-major lower Cholesky factor of the correlation matrix.
    factor: Vec<f64>,
}

fn correlation_matrix(measurements: usize, entry: impl Fn(usize, usize) -> f64) -> Vec<f64> {
    let mut matrix = vec![0.0; measurements * measurements];
    for i in 0..measurements {
        for j in 0..measurements {
            matrix[i * measurements + j] = if i == j { 1.0 } else { entry(i, j) };
        }
    }
    matrix
}

impl Covariance {
    pub(super) fn build(structure: &OutcomeStructure) -> Option<Self> {
        match structure {
            OutcomeStructure::RepeatedMeasures {
                measurements,
                variance,
                correlation: Some(correlation),
            } => {
                let m = *measurements as usize;
                let (dependence, matrix) = match *correlation {
                    WithinUnitCorrelation::CompoundSymmetric { rho } => {
                        (Dependence::CompoundSymmetric(rho), correlation_matrix(m, |_, _| rho))
                    }
                    WithinUnitCorrelation::Autoregressive { rho } => (
                        Dependence::Autoregressive(rho),
                        correlation_matrix(m, |i, j| rho.powi(i.abs_diff(j) as i32)),
                    ),
                };
                let factor = crate::gram::cholesky_spd(&matrix, m)?;
                Some(Self {
                    measurements: m,
                    marginal_variance: *variance,
                    dependence,
                    correlation: matrix,
                    factor,
                })
            }
            OutcomeStructure::Hierarchical {
                members,
                between_variance: Some(between),
                within_variance: Some(within),
            } => {
                let m = *members as usize;
                let total = between + within;
                let icc = between / total;
                let matrix = correlation_matrix(m, |_, _| icc);
                let factor = crate::gram::cholesky_spd(&matrix, m)?;
                Some(Self {
                    measurements: m,
                    marginal_variance: total,
                    dependence: Dependence::RandomEffect { between: *between, within: *within },
                    correlation: matrix,
                    factor,
                })
            }
            _ => None,
        }
    }

    /// Weights that reduce a unit's measurements to its summary; sum to one.
    fn weights(&self, summary: UnitSummary) -> Vec<f64> {
        let m = self.measurements;
        match (summary, &self.dependence) {
            (UnitSummary::Gls, Dependence::Autoregressive(rho)) => {
                let denominator = 2.0 + (m as f64 - 2.0) * (1.0 - rho);
                (0..m)
                    .map(|j| {
                        if j == 0 || j + 1 == m {
                            1.0 / denominator
                        } else {
                            (1.0 - rho) / denominator
                        }
                    })
                    .collect()
            }
            _ => vec![1.0 / m as f64; m],
        }
    }

    pub(super) fn reduce(&self, values: &[f64], summary: UnitSummary) -> f64 {
        self.weights(summary).iter().zip(values).map(|(w, v)| w * v).sum()
    }

    /// Variance of the unit summary: `marginal * w' R w`.
    pub(super) fn summary_variance(&self, summary: UnitSummary) -> f64 {
        let w = self.weights(summary);
        let m = self.measurements;
        let mut quadratic = 0.0;
        for i in 0..m {
            for j in 0..m {
                quadratic += w[i] * self.correlation[i * m + j] * w[j];
            }
        }
        self.marginal_variance * quadratic
    }

    pub(super) fn draw(&self, rng: &mut CausalRng, mean: f64) -> Vec<f64> {
        let m = self.measurements;
        if let Dependence::RandomEffect { between, within } = &self.dependence {
            let effect = between.sqrt() * standard_normal(rng);
            let sd = within.sqrt();
            (0..m).map(|_| mean + effect + sd * standard_normal(rng)).collect()
        } else {
            let z: Vec<f64> = (0..m).map(|_| standard_normal(rng)).collect();
            let sd = self.marginal_variance.sqrt();
            (0..m)
                .map(|i| {
                    let correlated: f64 = (0..=i).map(|k| self.factor[i * m + k] * z[k]).sum();
                    mean + sd * correlated
                })
                .collect()
        }
    }

    pub(super) fn is_independent(&self) -> bool {
        match self.dependence {
            Dependence::CompoundSymmetric(rho) | Dependence::Autoregressive(rho) => {
                rho.to_bits() << 1 == 0
            }
            Dependence::RandomEffect { between, .. } => between.to_bits() << 1 == 0,
        }
    }
}

/// Weights of the generalized-least-squares unit summary under the declared dependence and its
/// exact variance, `None` for an incomplete or inadmissible structure. The summary is the
/// sufficient statistic for a common mean of a unit's measurements, which is what the posterior
/// of [`super::super::repeated_measures_posterior`] conditions on.
pub(in crate::adaptive_inference) fn gls_unit_summary(
    structure: &OutcomeStructure,
) -> Option<(Vec<f64>, f64)> {
    let covariance = Covariance::build(structure)?;
    Some((covariance.weights(UnitSummary::Gls), covariance.summary_variance(UnitSummary::Gls)))
}
