//! Planned nuisance-fit counts, derived from the declared configuration of a route (2.2 cost
//! checks).
//!
//! A count here is read off the configuration the route executes: the penalty grid and inner
//! folds of a ridge propensity, and the folds, orderings and component count of a factorized
//! joint cell. It is a planning figure, exact when every stratum has rows of both classes and
//! otherwise an upper bound (a stratum with no training row, or only one class, is skipped
//! or fit as a constant), and it is never a runtime guarantee.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use crate::error::EstimationError;
use crate::joint_cell_factorized::{FactorizedJointConfig, prefix_mask, validate_orderings};
use crate::propensity::{NuisanceFallback, PropensityNuisance, PropensityPenalty, RidgeTuning};

fn count(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

impl RidgeTuning {
    /// Penalties one outer fold evaluates: the declared grid.
    #[must_use]
    pub fn planned_penalties(&self) -> u64 {
        count(self.lambdas().len())
    }

    /// Fits one outer fold's penalty selection plans, an upper bound: every penalty is fit on
    /// each inner training split (`inner_folds`), and once more on the whole training set when
    /// its inner loss improves on the best so far.
    #[must_use]
    pub fn planned_fits_per_outer_fold(&self) -> u64 {
        self.planned_penalties().saturating_mul(count(self.inner_folds()).saturating_add(1))
    }
}

impl PropensityNuisance {
    /// Propensity fits one outer cross-fit fold plans, an upper bound: one for the unpenalized
    /// logistic, the penalty selection's for a ridge or lasso grid
    /// ([`RidgeTuning::planned_fits_per_outer_fold`]), plus the fallback destination's
    /// selection when a ridge or lasso fallback is declared (it runs only when the GLM fit
    /// fails). A declared ML fallback is closed and adds no fit.
    #[must_use]
    pub fn planned_fits_per_fold(&self) -> u64 {
        let primary = match self.penalty() {
            PropensityPenalty::None => 1,
            PropensityPenalty::RidgeLogistic(tuning) | PropensityPenalty::Lasso(tuning) => {
                tuning.planned_fits_per_outer_fold()
            }
        };
        let fallback = match self.fallback() {
            NuisanceFallback::None | NuisanceFallback::Ml => 0,
            NuisanceFallback::RidgeLogistic(tuning) | NuisanceFallback::Lasso(tuning) => {
                tuning.planned_fits_per_outer_fold()
            }
        };
        primary.saturating_add(fallback)
    }

    /// Whether a ridge or lasso fallback is declared, so a failed GLM fit re-runs the whole
    /// cross-fitted route (outcome models included) with the penalized propensity.
    #[must_use]
    pub fn reruns_on_fallback(&self) -> bool {
        matches!(self.fallback(), NuisanceFallback::RidgeLogistic(_) | NuisanceFallback::Lasso(_))
    }
}

/// Planned nuisance fits of one factorized joint-cell fit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JointFitPlan {
    /// Cross-fit folds.
    pub folds: u64,
    /// Cells of the family (`2^components`).
    pub cells: u64,
    /// Orderings the family is estimated under.
    pub orderings: u64,
    /// Distinct prefix-stratum conditionals of one fold across every ordering: for each
    /// distinct (set of earlier components, target component) pair, one conditional per
    /// value of the earlier components. Conditionals are shared between orderings.
    pub conditional_strata: u64,
    /// `folds * conditional_strata`, an upper bound (an empty or one-class stratum is not fit).
    pub conditional_fits: u64,
    /// Fits of those conditionals: one each under a declared learner, the penalty selection's
    /// fits ([`RidgeTuning::planned_fits_per_outer_fold`]) each on the ridge route.
    pub propensity_fits: u64,
    /// Per-cell outcome models: one per cell per fold, an upper bound (an empty cell is not fit).
    pub outcome_fits: u64,
}

impl JointFitPlan {
    /// `propensity_fits + outcome_fits`.
    #[must_use]
    pub const fn nuisance_fits(&self) -> u64 {
        self.propensity_fits.saturating_add(self.outcome_fits)
    }
}

impl FactorizedJointConfig {
    /// The nuisance fits a [`crate::fit_factorized_joint_cells`] call with this configuration,
    /// `components` treatment components and these `orderings` plans.
    ///
    /// # Errors
    ///
    /// The refusal the fit itself raises for an invalid configuration or ordering list (no
    /// fit is planned for a declaration the route refuses before any fit).
    pub fn planned_fits(
        &self,
        components: usize,
        orderings: &[Vec<usize>],
    ) -> Result<JointFitPlan, EstimationError> {
        self.validate()?;
        validate_orderings(components, orderings)?;
        let strata: BTreeSet<(u32, usize)> = orderings
            .iter()
            .flat_map(|ordering| {
                (0..ordering.len()).map(move |j| (prefix_mask(ordering, j), ordering[j]))
            })
            .collect();
        let conditional_strata: u64 =
            strata.iter().map(|(mask, _)| 1_u64 << mask.count_ones()).sum();
        let folds = count(self.folds);
        let conditional_fits = folds.saturating_mul(conditional_strata);
        let per_conditional = match self.learner {
            Some(_) => 1,
            None => self.tuning.planned_fits_per_outer_fold(),
        };
        let cells = 1_u64 << components;
        Ok(JointFitPlan {
            folds,
            cells,
            orderings: count(orderings.len()),
            conditional_strata,
            conditional_fits,
            propensity_fits: conditional_fits.saturating_mul(per_conditional),
            outcome_fits: cells.saturating_mul(folds),
        })
    }
}
