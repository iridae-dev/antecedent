//! Kernel-smoothed dose-response transport query (2.2B cell X4).
//! SPDX-License-Identifier: MIT OR Apache-2.0
//!
//! The estimand is `psi_h(a) = E_target[ integral K_h(a - t) E(Y | X, A=t, S=1) dt ]` for
//! each declared grid dose `a`, one declared bandwidth `h` and the declared kernel `K`.
//! The bandwidth and kernel are part of the estimand's identity: a different `h` is a
//! different target, never a tuning choice. This query is distinct from the conditional
//! [`super::ContinuousDoseResponseQuery`] (group-level exchangeability, no transport) and
//! from a stochastic intervention: `psi_h(a)` equals the target mean under the stochastic
//! dose `T ~ K_h(a - .)` independent of `X`, but the query type, the declared smoothing
//! target and the refusals of point, stochastic, coarsened and derivative targets are
//! what keep the routes apart.

use super::TransportQuery;
use super::{ContinuousDomain, GridSpec, QueryError, ResponseFunctional, ResponseQuery};
use crate::ids::VariableId;
use std::sync::Arc;

/// The smoothing kernel of the estimand. Only one is supported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmoothingKernel {
    /// `K(u) = 0.75 (1 - u^2)` on `[-1, 1]` (second moment `1/5`).
    Epanechnikov,
}

impl SmoothingKernel {
    /// Stable kernel name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Epanechnikov => "epanechnikov",
        }
    }

    /// The kernel density at `u`; zero outside `[-1, 1]`.
    #[must_use]
    pub fn density(self, u: f64) -> f64 {
        match self {
            Self::Epanechnikov => {
                if u.abs() <= 1.0 {
                    0.75 * (1.0 - u * u)
                } else {
                    0.0
                }
            }
        }
    }
}

/// A fixed-bandwidth smoothed dose-response grid transported to a target population.
#[derive(Clone, Debug, PartialEq)]
pub struct SmoothedDoseTransportQuery {
    /// Outcome column.
    pub outcome: VariableId,
    /// Randomized continuous source dose; the single source experiment.
    pub dose: VariableId,
    /// Source (trial) population key.
    pub source_population: Arc<str>,
    /// Target population key.
    pub target_population: Arc<str>,
    /// Grid doses in the caller's order (distinct, finite).
    pub grid: Arc<[f64]>,
    /// Kernel bandwidth `h > 0`; part of the estimand.
    pub bandwidth: f64,
    /// Smoothing kernel; part of the estimand.
    pub kernel: SmoothingKernel,
    /// Declared randomized dose support `(lo, hi)`, `lo < hi`.
    pub dose_support: (f64, f64),
    /// How the known conditional dose density was obtained (`known`, or a refused
    /// estimated provenance).
    pub density_provenance: Arc<str>,
}

impl SmoothedDoseTransportQuery {
    /// Validate the structural shape of the query. Support, bounds and provenance are
    /// estimator refusals, not query errors.
    ///
    /// # Errors
    ///
    /// [`QueryError::InvalidResponse`] for equal dose and outcome coordinates, an empty
    /// grid, a non-finite or repeated grid dose, a non-positive or non-finite bandwidth,
    /// an empty or non-finite dose support, blank or equal population keys, or a blank
    /// density provenance.
    pub fn validate(&self) -> Result<(), QueryError> {
        let (lo, hi) = self.dose_support;
        let mut sorted = self.grid.to_vec();
        sorted.sort_by(f64::total_cmp);
        if self.outcome == self.dose
            || self.grid.is_empty()
            || self.grid.iter().any(|a| !a.is_finite())
            || sorted.windows(2).any(|pair| pair[0].total_cmp(&pair[1]).is_eq())
            || !self.bandwidth.is_finite()
            || self.bandwidth <= 0.0
            || !lo.is_finite()
            || !hi.is_finite()
            || lo >= hi
            || self.source_population.trim().is_empty()
            || self.target_population.trim().is_empty()
            || self.source_population == self.target_population
            || self.density_provenance.trim().is_empty()
        {
            return Err(QueryError::InvalidResponse(
                "a smoothed dose-response transport query needs distinct dose and outcome \
                 coordinates, a non-empty grid of distinct finite doses, a positive finite \
                 bandwidth, a finite dose support lo < hi, distinct population keys and a \
                 density provenance"
                    .into(),
            ));
        }
        Ok(())
    }

    /// The transport query the certificate is derived from: the mean response curve of
    /// the outcome over the declared dose support, with the dose as the one source
    /// experiment.
    #[must_use]
    pub fn transport_query(&self) -> TransportQuery {
        let (lo, hi) = self.dose_support;
        TransportQuery::new(
            ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome: self.outcome,
                treatment: ContinuousDomain::new(self.dose, GridSpec::Values(Arc::from([lo, hi]))),
            }),
            Arc::clone(&self.source_population),
            Arc::clone(&self.target_population),
            [self.dose],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query() -> SmoothedDoseTransportQuery {
        SmoothedDoseTransportQuery {
            outcome: VariableId::from_raw(2),
            dose: VariableId::from_raw(1),
            source_population: Arc::from("trial"),
            target_population: Arc::from("target"),
            grid: Arc::from([2.0, 1.0]),
            bandwidth: 0.5,
            kernel: SmoothingKernel::Epanechnikov,
            dose_support: (0.0, 4.0),
            density_provenance: Arc::from("known"),
        }
    }

    #[test]
    fn structural_validation_accepts_any_grid_order_and_refuses_malformed_queries() {
        assert!(query().validate().is_ok());
        let cases = [
            SmoothedDoseTransportQuery { grid: Arc::from([]), ..query() },
            SmoothedDoseTransportQuery { grid: Arc::from([1.0, 1.0]), ..query() },
            SmoothedDoseTransportQuery { grid: Arc::from([f64::NAN]), ..query() },
            SmoothedDoseTransportQuery { bandwidth: 0.0, ..query() },
            SmoothedDoseTransportQuery { bandwidth: f64::INFINITY, ..query() },
            SmoothedDoseTransportQuery { dose_support: (4.0, 0.0), ..query() },
            SmoothedDoseTransportQuery { dose: VariableId::from_raw(2), ..query() },
            SmoothedDoseTransportQuery { target_population: Arc::from("trial"), ..query() },
            SmoothedDoseTransportQuery { density_provenance: Arc::from(" "), ..query() },
        ];
        for case in cases {
            assert!(case.validate().is_err(), "{case:?}");
        }
        let transport = query().transport_query();
        assert!(transport.validate().is_ok());
        assert_eq!(&*transport.source_experiments, &[VariableId::from_raw(1)]);
    }

    #[test]
    fn the_epanechnikov_kernel_is_a_unit_density_with_second_moment_one_fifth() {
        let k = SmoothingKernel::Epanechnikov;
        assert_eq!(k.name(), "epanechnikov");
        assert!(k.density(1.5).abs() < f64::EPSILON && (k.density(0.0) - 0.75).abs() < 1e-15);
        let steps = 200_000;
        let du = 2.0 / f64::from(steps);
        let (mut mass, mut second) = (0.0, 0.0);
        for i in 0..steps {
            let u = -1.0 + (f64::from(i) + 0.5) * du;
            mass += k.density(u) * du;
            second += u * u * k.density(u) * du;
        }
        assert!((mass - 1.0).abs() < 1e-8 && (second - 0.2).abs() < 1e-8);
    }
}
