//! Overlap-rule diagnostics (DESIGN.md §18.2).
//!
//! Distinct from [`crate::OverlapRefuter`]: that check reports propensity range / ESS.
//! This module evaluates a **trimming / common-support rule** — whether enough mass remains
//! after excluding extreme propensities under a declared rule (Crump-style fixed band or
//! minimum retained support fraction).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]

use std::sync::Arc;

use causal_estimate::{OverlapPolicy, OverlapReport};
use causal_stats::GlmOptions;

use crate::common::{RefutationProblem, RefutationReport, diagnostic_overlap_report};
use crate::error::ValidationError;

/// Common-support / trimming-rule assessment.
#[derive(Clone, Debug)]
pub struct OverlapRuleRefuter {
    /// Propensity band `[rule_eps, 1 - rule_eps]` retained by the rule.
    pub rule_eps: f64,
    /// Minimum fraction of units that must remain inside the band.
    pub min_retained_fraction: f64,
    /// GLM options for diagnostic-only propensity when the original estimate has none.
    pub glm_options: GlmOptions,
}

impl Default for OverlapRuleRefuter {
    fn default() -> Self {
        Self::new()
    }
}

impl OverlapRuleRefuter {
    /// Defaults: `rule_eps = 0.1` (Crump-style), `min_retained_fraction = 0.5`.
    #[must_use]
    pub fn new() -> Self {
        Self { rule_eps: 0.1, min_retained_fraction: 0.5, glm_options: GlmOptions::default() }
    }

    /// Run the overlap-rule diagnostic.
    ///
    /// # Errors
    ///
    /// Data or GLM failures while building a diagnostic-only propensity fit.
    pub fn refute(
        &self,
        problem: &RefutationProblem<'_>,
    ) -> Result<RefutationReport, ValidationError> {
        let eps = self.rule_eps.clamp(1e-6, 0.49);
        let report = match &problem.original.overlap_report {
            Some(r) => r.clone(),
            None => self.diagnostic_report(problem, eps)?,
        };
        // Prefer the §14.3 support field when the report's band matches the declared rule;
        // if the observed range sits fully inside the band, retention is exactly 1. A reused
        // report evaluated at a *different* clip must not stand in for the rule's band — its
        // support figure answers a different question — so refit a diagnostic propensity at
        // the rule's band instead.
        let band_matches = report.clip.is_some_and(|clip| (clip - eps).abs() < 1e-12);
        let retained = if band_matches {
            report.target_population_support
        } else if report.propensity_min >= eps && report.propensity_max <= 1.0 - eps {
            1.0
        } else {
            self.diagnostic_report(problem, eps)?.target_population_support
        };
        let passed = retained >= self.min_retained_fraction;
        Ok(RefutationReport {
            refuter: Arc::from("overlap.rule"),
            original_ate: problem.original.ate,
            refuted_ate: problem.original.ate,
            comparison: 1.0 - retained,
            informative: true,
            passed,
            failure_condition: if passed {
                None
            } else {
                Some(Arc::from(format!(
                    "overlap rule eps={eps}: retained_fraction={retained} < min {}",
                    self.min_retained_fraction
                )))
            },
            replicates: 0,
        })
    }

    fn diagnostic_report(
        &self,
        problem: &RefutationProblem<'_>,
        eps: f64,
    ) -> Result<OverlapReport, ValidationError> {
        diagnostic_overlap_report(
            problem,
            &self.glm_options,
            OverlapPolicy::RequireDiagnostics { clip: Some(eps), trim: None },
        )
    }
}

#[cfg(test)]
mod tests {
    use causal_core::{AssumptionSet, AverageEffectQuery, ExecutionContext, VariableId};
    use causal_estimate::{EstimationWorkspace, LinearAdjustmentAte};

    use super::*;
    use crate::common::RefutationProblem;

    // Reuse the shared toy from lib tests via a minimal inline SCM.
    fn toy() -> (causal_data::TabularData, causal_identify::IdentifiedEstimand) {
        use std::sync::Arc;

        use causal_core::{
            CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
        };
        use causal_data::{
            Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
        };
        use causal_expr::ExprId;
        use causal_identify::IdentifiedEstimand;

        let n = 200usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "t",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "z",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let z: Vec<f64> = (0..n).map(|i| (i as f64) / n as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * t[i] + z[i]).collect();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(z),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([VariableId::from_raw(2)]),
            ExprId::from_raw(0),
        );
        (TabularData::new(storage), estimand)
    }

    #[test]
    fn overlap_rule_passes_on_balanced_toy() {
        let (data, estimand) = toy();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        let problem = RefutationProblem {
            data: &data,
            estimand: &estimand,
            query: &query,
            original: &original,
            estimator: Some("linear.adjustment.ate"),
        };
        let report = OverlapRuleRefuter::new().refute(&problem).unwrap();
        assert_eq!(report.refuter.as_ref(), "overlap.rule");
        assert!(report.informative);
        assert!(report.passed, "comparison={}", report.comparison);
    }
}
