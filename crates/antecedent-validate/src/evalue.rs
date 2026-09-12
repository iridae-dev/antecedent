//! E-value sensitivity analysis (`VanderWeele` & Ding, 2017).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::common::{RefutationProblem, RefutationReport, complete_case_rows, masked_sample_sd};
use crate::error::ValidationError;

/// Library-selected pass threshold of E ≥ 2, not a universal robustness criterion.
/// The original E-value formulation reports it as a
/// continuous diagnostic without a pass/fail gate; this library uses 2.0 so `ValidationSuite` verdicts are not
/// vacuous (the formula always yields E ≥ 1, so threshold 1.0 would pass every estimate,
/// including a true null).
pub const DEFAULT_EVALUE_THRESHOLD: f64 = 2.0;

/// E-value for the point estimate: the minimum strength of association, on the risk-ratio
/// scale, that an unmeasured confounder would need with both treatment and outcome to fully
/// explain away the observed effect.
///
/// For continuous outcomes this uses the `VanderWeele`/Ding approximate conversion of the
/// standardized mean difference `d = ATE / SD(Y)` to a risk ratio via `RR = exp(0.91 d)`,
/// then the standard E-value formula `E = RR + sqrt(RR (RR − 1))` (inverted first if `RR < 1`).
#[derive(Clone, Debug)]
pub struct EValue {
    /// Pass if the computed E-value is at least this large.
    ///
    /// Default [`DEFAULT_EVALUE_THRESHOLD`] (2.0) is a library convention. Override via
    /// [`EValue::with_threshold`] when a different convention is needed; the E-value itself
    /// is always reported in [`RefutationReport::comparison`] regardless of the gate.
    pub threshold: f64,
}

impl Default for EValue {
    fn default() -> Self {
        Self::new()
    }
}

impl EValue {
    /// Default threshold [`DEFAULT_EVALUE_THRESHOLD`] (2.0, a library convention).
    #[must_use]
    pub fn new() -> Self {
        Self { threshold: DEFAULT_EVALUE_THRESHOLD }
    }

    /// Explicit pass threshold (for report-only use, set `threshold = 0.0` or ignore
    /// `passed` and read [`RefutationReport::comparison`]).
    #[must_use]
    pub fn with_threshold(threshold: f64) -> Self {
        Self { threshold }
    }

    /// Compute the E-value for `problem.original.ate`.
    ///
    /// # Errors
    ///
    /// The outcome has fewer than 2 valid rows or zero variance.
    pub fn refute(
        &self,
        problem: &RefutationProblem<'_>,
    ) -> Result<RefutationReport, ValidationError> {
        let sd_y = if problem.effect_refit.is_some() {
            let (mask, _) = complete_case_rows(problem.data, &[problem.outcome()])?;
            masked_sample_sd(problem.data, problem.outcome(), &mask)?
        } else if problem.temporal.is_some() {
            let prep = crate::common::temporal_diagnostic_design(problem)?;
            crate::common::sample_sd(&prep.design.outcome)
        } else {
            let mut ids = vec![problem.treatment(), problem.outcome()];
            ids.extend_from_slice(&problem.estimand.adjustment_set);
            let (mask, _valid) = complete_case_rows(problem.data, &ids)?;
            masked_sample_sd(problem.data, problem.outcome(), &mask)?
        };
        if !(sd_y.is_finite() && sd_y > 0.0) {
            return Err(ValidationError::NotApplicable {
                message: "e-value requires a finite, positive outcome standard deviation",
            });
        }
        let d = problem.original.ate / sd_y;
        let rr = (0.91 * d.abs()).exp();
        let e_value = e_value_from_risk_ratio(rr);
        let passed = e_value >= self.threshold;
        Ok(RefutationReport {
            refuter: Arc::from("sensitivity.evalue"),
            original_ate: problem.original.ate,
            refuted_ate: problem.original.ate,
            comparison: e_value,
            informative: true,
            passed,
            failure_condition: if passed {
                None
            } else {
                Some(Arc::from(format!("e-value {e_value} below threshold {}", self.threshold)))
            },
            replicates: 0,
        })
    }
}

fn e_value_from_risk_ratio(rr: f64) -> f64 {
    let rr = if rr >= 1.0 { rr } else { 1.0 / rr };
    rr + rr.sqrt() * (rr - 1.0).sqrt()
}

#[cfg(test)]
mod tests {
    use super::e_value_from_risk_ratio;

    #[test]
    fn large_finite_risk_ratio_does_not_overflow_intermediate_product() {
        let value = e_value_from_risk_ratio(1e200);
        assert!(value.is_finite());
        assert!((value / 2e200 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn protective_and_harmful_risk_ratios_have_same_evalue() {
        assert!((e_value_from_risk_ratio(0.5) - e_value_from_risk_ratio(2.0)).abs() < 1e-12);
    }
}
