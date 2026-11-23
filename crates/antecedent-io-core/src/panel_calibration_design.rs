//! Match frozen panel queries to their measured calibration designs.

use crate::query_wire::{SequentialContrastQueryWire, SyntheticControlQueryWire};

impl SequentialContrastQueryWire {
    /// Design used by the interval studies conditional on passing the fitted overlap check.
    #[must_use]
    pub fn matches_wide_overlap_study(&self) -> bool {
        self.active == [1, 1]
            && self.control == [0, 0]
            && self.baseline.len() == 1
            && self.intermediate.len() == 1
            && self.min_probability.to_bits() == 0.05_f64.to_bits()
            && self.fold_count == 5
            && self.fold_of_row.len() >= 5
            && self.fold_of_row.iter().enumerate().all(|(i, &fold)| (i % 5) == fold as usize)
    }
}

impl SyntheticControlQueryWire {
    /// Calibration identity of the single-treated-unit Bayesian average. The grid
    /// coordinate is observed pre-treatment periods. Donors and horizon are fixed
    /// within one grid, and the predictive path has a distinct construction.
    #[must_use]
    pub fn average_calibration_design(&self) -> Option<(String, u64)> {
        use std::collections::BTreeSet;
        if self.difference_in_differences
            || self.uniform_unit_randomization
            || self.sharp_null_effect.is_some()
            || self.augmentation_ridge.is_some()
            || self.observed_window_support
            || self.units.len() != self.periods.len()
        {
            return None;
        }
        let construction = match &self.rollout {
            None => "posterior_average",
            Some(rollout)
                if rollout.interval == "conformal"
                    && matches!(rollout.adoptions.as_slice(), [(unit, period)]
                        if unit == &self.treated_unit && *period == self.intervention_period) =>
            {
                "predictive_path_average"
            }
            Some(_) => return None,
        };
        let donors: BTreeSet<_> = self.units.iter().filter(|u| *u != &self.treated_unit).collect();
        let periods: BTreeSet<_> = self
            .units
            .iter()
            .zip(&self.periods)
            .filter_map(|(unit, period)| (unit == &self.treated_unit).then_some(*period))
            .collect();
        let pre = periods.range(..self.intervention_period).count();
        let post = periods.len() - pre;
        if pre == 0 || post == 0 || donors.is_empty() {
            return None;
        }
        Some((
            format!("synthetic_control/{construction}+donors={}+post_periods={post}", donors.len()),
            pre as u64,
        ))
    }
}
