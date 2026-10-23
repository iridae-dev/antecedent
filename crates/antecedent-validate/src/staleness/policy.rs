//! The declared policy a staleness diagnostic is judged by.
//!
//! Every threshold, test level, multiplicity rule, minimum sample size and random-stream seed
//! that can change a finding is a field here, and every field enters
//! [`StalenessPolicy::identity`]. The policy identity is part of the report identity, so two
//! reports under different policies never compare equal.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::CanonicalWriter;

use super::StalenessError;

const POLICY_TAG: &str = "antecedent.validate.staleness.policy.v1";

/// How the p-values of the test-based diagnostics are adjusted for multiplicity.
///
/// Every test-based diagnostic of one assessment (per-column shift, joint shift, support,
/// level disappearance, missingness, period drift, prior conflict) is one family. Holm controls
/// the family-wise error rate under any dependence; Benjamini–Hochberg controls the false
/// discovery rate (and the family-wise rate when no shift exists at all).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Multiplicity {
    /// Holm step-down (family-wise error rate).
    Holm,
    /// Benjamini–Hochberg step-up (false discovery rate).
    BenjaminiHochberg,
}

impl Multiplicity {
    /// Stable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Holm => "holm",
            Self::BenjaminiHochberg => "benjamini_hochberg",
        }
    }

    /// Parse [`Self::label`].
    #[must_use]
    pub fn parse(label: &str) -> Option<Self> {
        match label {
            "holm" => Some(Self::Holm),
            "benjamini_hochberg" | "bh" => Some(Self::BenjaminiHochberg),
            _ => None,
        }
    }
}

/// Thresholds, tests and sample-size floors of a staleness diagnostic.
#[derive(Clone, Debug, PartialEq)]
pub struct StalenessPolicy {
    /// Level of the test-based diagnostics after multiplicity adjustment.
    pub alpha: f64,
    /// Multiplicity control across the test-based diagnostics.
    pub multiplicity: Multiplicity,
    /// Smallest absolute standardized mean difference that counts as a shift.
    pub smd_threshold: f64,
    /// Smallest population stability index (binned density-ratio divergence) that counts as a
    /// shift.
    pub psi_threshold: f64,
    /// Smallest Mahalanobis distance between the covariate means that counts as a joint shift.
    pub mahalanobis_threshold: f64,
    /// Quantile bins per numeric column of the baseline.
    pub bins: usize,
    /// Rows of the uniform covariate subsample on each side of the energy test.
    pub subsample: usize,
    /// Permutations of the energy test.
    pub permutations: u32,
    /// Seed of the keyed random streams (subsampling and permutations).
    pub seed: u64,
    /// Positivity margin: a propensity outside `[epsilon, 1 - epsilon]` is extreme.
    pub positivity_epsilon: f64,
    /// Largest fraction of extreme propensities that keeps positivity.
    pub max_extreme_fraction: f64,
    /// Smallest ratio of the new to the recorded effective-sample-size fraction.
    pub min_ess_ratio_change: f64,
    /// Largest weighted standardized mean difference that keeps balance.
    pub balance_threshold: f64,
    /// Largest condition number (lower bound) of the adjustment covariates' correlation.
    pub max_condition: f64,
    /// Smallest fraction of rows complete on the plan's columns.
    pub min_complete_fraction: f64,
    /// Largest fraction of new rows outside the recorded range that is tolerated.
    pub support_tolerance: f64,
    /// Smallest count of rows at a level the baseline never saw that counts as a new level.
    pub min_level_rows: u64,
    /// Smallest absolute change of a missingness rate (or total-variation change of the
    /// missingness patterns) that counts.
    pub missingness_threshold: f64,
    /// Smallest number of new rows complete on the plan's columns.
    pub min_rows: u64,
    /// Smallest number of new rows in each treatment arm.
    pub min_rows_per_arm: u64,
    /// Smallest number of rows of a period that enters the period-drift test.
    pub min_rows_per_period: u64,
    /// Column whose values define observation periods, when declared.
    pub period_column: Option<String>,
    /// Prior variance (in units of the noise variance) of every outcome-regression coefficient
    /// of the reference conjugate prior used by the prior-likelihood check.
    pub prior_variance: f64,
    /// Shape of the reference inverse-gamma prior of the noise variance.
    pub prior_shape: f64,
    /// Rate of the reference inverse-gamma prior of the noise variance.
    pub prior_rate: f64,
}

impl Default for StalenessPolicy {
    fn default() -> Self {
        Self {
            alpha: 0.01,
            multiplicity: Multiplicity::Holm,
            smd_threshold: 0.1,
            psi_threshold: 0.1,
            mahalanobis_threshold: 0.2,
            bins: 10,
            subsample: 200,
            permutations: 199,
            seed: 0x5EED_57A1_E000_0001,
            positivity_epsilon: 0.01,
            max_extreme_fraction: 0.01,
            min_ess_ratio_change: 0.5,
            balance_threshold: 0.1,
            max_condition: 1e8,
            min_complete_fraction: 0.5,
            support_tolerance: 0.01,
            min_level_rows: 1,
            missingness_threshold: 0.05,
            min_rows: 30,
            min_rows_per_arm: 10,
            min_rows_per_period: 5,
            period_column: None,
            prior_variance: 100.0,
            prior_shape: 0.01,
            prior_rate: 0.01,
        }
    }
}

fn unit_open(value: f64) -> bool {
    value.is_finite() && value > 0.0 && value < 1.0
}

fn non_negative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

impl StalenessPolicy {
    /// Refuse a policy whose numbers cannot judge a finding.
    ///
    /// # Errors
    /// [`StalenessError::Policy`] naming the first invalid field.
    pub fn validate(&self) -> Result<(), StalenessError> {
        let checks: [(bool, &'static str); 18] = [
            (unit_open(self.alpha), "alpha must lie in (0, 1)"),
            (non_negative(self.smd_threshold), "smd_threshold must be finite and nonnegative"),
            (non_negative(self.psi_threshold), "psi_threshold must be finite and nonnegative"),
            (
                non_negative(self.mahalanobis_threshold),
                "mahalanobis_threshold must be finite and nonnegative",
            ),
            ((2..=1000).contains(&self.bins), "bins must lie in [2, 1000]"),
            ((2..=5000).contains(&self.subsample), "subsample must lie in [2, 5000]"),
            ((1..=100_000).contains(&self.permutations), "permutations must lie in [1, 100000]"),
            (
                self.positivity_epsilon.is_finite()
                    && self.positivity_epsilon > 0.0
                    && self.positivity_epsilon < 0.5,
                "positivity_epsilon must lie in (0, 0.5)",
            ),
            (
                (0.0..1.0).contains(&self.max_extreme_fraction),
                "max_extreme_fraction must lie in [0, 1)",
            ),
            (unit_open(self.min_ess_ratio_change), "min_ess_ratio_change must lie in (0, 1)"),
            (positive(self.balance_threshold), "balance_threshold must be finite and positive"),
            (
                self.max_condition.is_finite() && self.max_condition >= 1.0,
                "max_condition must be finite and at least 1",
            ),
            (
                (0.0..=1.0).contains(&self.min_complete_fraction),
                "min_complete_fraction must lie in [0, 1]",
            ),
            ((0.0..1.0).contains(&self.support_tolerance), "support_tolerance must lie in [0, 1)"),
            (unit_open(self.missingness_threshold), "missingness_threshold must lie in (0, 1)"),
            (self.min_rows >= 2, "min_rows must be at least 2"),
            (self.min_rows_per_period >= 2, "min_rows_per_period must be at least 2"),
            (
                positive(self.prior_variance)
                    && positive(self.prior_shape)
                    && positive(self.prior_rate),
                "prior_variance, prior_shape and prior_rate must be finite and positive",
            ),
        ];
        for (ok, message) in checks {
            if !ok {
                return Err(StalenessError::Policy(message));
            }
        }
        if self.min_level_rows == 0 {
            return Err(StalenessError::Policy("min_level_rows must be at least 1"));
        }
        if self.period_column.as_deref().is_some_and(str::is_empty) {
            return Err(StalenessError::Policy("period_column must not be empty"));
        }
        Ok(())
    }

    /// Canonical identity of every field (lowercase hex).
    #[must_use]
    pub fn identity(&self) -> String {
        let mut writer = CanonicalWriter::new(POLICY_TAG);
        writer
            .real(self.alpha)
            .text(self.multiplicity.label())
            .real(self.smd_threshold)
            .real(self.psi_threshold)
            .real(self.mahalanobis_threshold)
            .word(self.bins as u64)
            .word(self.subsample as u64)
            .word(u64::from(self.permutations))
            .word(self.seed)
            .real(self.positivity_epsilon)
            .real(self.max_extreme_fraction)
            .real(self.min_ess_ratio_change)
            .real(self.balance_threshold)
            .real(self.max_condition)
            .real(self.min_complete_fraction)
            .real(self.support_tolerance)
            .word(self.min_level_rows)
            .real(self.missingness_threshold)
            .word(self.min_rows)
            .word(self.min_rows_per_arm)
            .word(self.min_rows_per_period)
            .opt_text(self.period_column.as_deref())
            .real(self.prior_variance)
            .real(self.prior_shape)
            .real(self.prior_rate);
        writer.finish_hex()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_field_enters_the_identity_and_invalid_policies_refuse() {
        let base = StalenessPolicy::default();
        base.validate().expect("default policy is valid");
        let id = base.identity();
        let mut alpha = base.clone();
        alpha.alpha = 0.05;
        assert_ne!(alpha.identity(), id);
        let mut multiplicity = base.clone();
        multiplicity.multiplicity = Multiplicity::BenjaminiHochberg;
        assert_ne!(multiplicity.identity(), id);
        let mut period = base.clone();
        period.period_column = Some("week".into());
        assert_ne!(period.identity(), id);
        let mut seed = base.clone();
        seed.seed += 1;
        assert_ne!(seed.identity(), id);
        let mut bad = base;
        bad.alpha = 1.5;
        assert!(matches!(bad.validate(), Err(StalenessError::Policy(_))));
    }
}
