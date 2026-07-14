//! Propensity-based estimators: weighting, stratification, and matching .
//!
//! All estimators here require propensity-based positivity diagnostics
//! ([`OverlapPolicy::RequireDiagnostics`]) — [`OverlapPolicy::ExplicitOverride`] is refused
//! because positivity is mandatory for propensity/matching methods (DESIGN.md §14.3).
//!
//! Bootstrap standard errors **refit the propensity model on every resample** rather than
//! reusing the point-estimate propensity scores. This is more expensive than reusing scores,
//! but it is the honest choice: it propagates first-stage estimation uncertainty into the
//! second-stage effect, which score-reuse would understate. [`causal_stats::PropensityWorkspace`]
//! scratch (IRLS design/Cholesky buffers) is reused across all replicates to keep the
//! per-replicate cost to a single GLM refit with no additional heap churn.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::manual_memcpy,
    clippy::needless_pass_by_value
)]

use std::sync::Arc;

use causal_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, TargetPopulation, VariableId,
};
use causal_data::TabularData;
use causal_expr::IdentifiedEstimand;
use causal_stats::{
    FaerBackend, GlmOptions, MatchingDistance, MatchingIndex, PropensityFit, PropensityWorkspace,
    fit_propensity,
};

use crate::adjustment::{EffectEstimate, intervention_f64};
use crate::error::EstimationError;
use crate::overlap::{OverlapPolicy, OverlapReport};
use crate::util::{sample_std, stats_err};

include!("prepare.rs");
include!("weighting.rs");
include!("stratification.rs");
include!("matching.rs");
include!("distance.rs");
#[cfg(test)]
#[allow(clippy::many_single_char_names, clippy::float_cmp)]
mod tests {
    use std::sync::Arc;

    use causal_core::{
        CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet,
        TargetPopulation, ValueType, VariableId,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use causal_expr::ExprId;
    use causal_expr::IdentifiedEstimand;

    use super::*;
    use crate::overlap::OverlapPolicy;

    /// Confounded SCM: `Z ~ N(0,1)`, `T ~ Bernoulli(logit(-0.5 + Z))`, `Y = 2T + Z + noise`.
    /// True ATE = 2.
    fn standard_normal(rng: &mut causal_core::CausalRng) -> f64 {
        // Box-Muller.
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn confounded_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
        let (t, y, z) = confounded_columns(n, seed);
        build_dataset(t, y, z)
    }

    fn confounded_columns(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut rng = ExecutionContext::for_tests(seed).rng.stream(0x1234_u64);

        let mut z = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let zi = standard_normal(&mut rng);
            let logit = -0.5 + zi;
            let p = 1.0 / (1.0 + (-logit).exp());
            let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
            let noise = standard_normal(&mut rng) * 0.5;
            z[i] = zi;
            t[i] = ti;
            y[i] = 2.0 * ti + zi + noise;
        }
        (t, y, z)
    }

    /// `confounded_scm` plus one extreme-propensity treated outlier: `z = -8` puts its raw
    /// propensity near 2e-4 (outside any reasonable trim band) while `y = 1000` wrecks any
    /// estimator that fails to exclude it.
    fn confounded_scm_with_outlier(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
        let (mut t, mut y, mut z) = confounded_columns(n, seed);
        t.push(1.0);
        y.push(1000.0);
        z.push(-8.0);
        build_dataset(t, y, z)
    }

    fn build_dataset(t: Vec<f64>, y: Vec<f64>, z: Vec<f64>) -> (TabularData, IdentifiedEstimand) {
        let n = t.len();
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

    fn ctx() -> ExecutionContext {
        ExecutionContext::for_tests(7)
    }

    /// Diagnostics-mandatory policy with an explicit trim band for the outlier tests.
    fn trim_overlap() -> OverlapPolicy {
        OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: Some(0.02) }
    }

    #[test]
    fn weighting_recovers_ate_two() {
        let (data, estimand) = confounded_scm(800, 1);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = PropensityWeighting { bootstrap_replicates: 30, ..PropensityWeighting::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = PropensityEstimationWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!((effect.ate - 2.0).abs() < 0.3, "ate={}", effect.ate);
        assert!(effect.se_bootstrap.is_some());
        assert!(effect.overlap_report.is_some());
    }

    #[test]
    fn weighting_att_target_population() {
        let (data, estimand) = confounded_scm(800, 2);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::Treated);
        let est = PropensityWeighting { bootstrap_replicates: 0, ..PropensityWeighting::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = PropensityEstimationWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!((effect.ate - 2.0).abs() < 0.4, "att={}", effect.ate);
    }

    #[test]
    fn weighting_rejects_explicit_override() {
        let (data, estimand) = confounded_scm(200, 3);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = PropensityWeighting {
            overlap: OverlapPolicy::ExplicitOverride,
            ..PropensityWeighting::new()
        };
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::Overlap { .. }));
    }

    #[test]
    fn stratification_recovers_ate_two() {
        let (data, estimand) = confounded_scm(800, 4);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = PropensityStratification {
            bootstrap_replicates: 30,
            ..PropensityStratification::new()
        };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = PropensityEstimationWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!((effect.ate - 2.0).abs() < 0.3, "ate={}", effect.ate);
        assert!(effect.se_bootstrap.is_some());
    }

    #[test]
    fn stratification_rejects_explicit_override() {
        let (data, estimand) = confounded_scm(200, 5);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = PropensityStratification {
            overlap: OverlapPolicy::ExplicitOverride,
            ..PropensityStratification::new()
        };
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::Overlap { .. }));
    }

    #[test]
    fn propensity_matching_recovers_att() {
        let (data, estimand) = confounded_scm(800, 6);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::Treated);
        let est = PropensityMatching { bootstrap_replicates: 30, ..PropensityMatching::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = PropensityEstimationWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!((effect.ate - 2.0).abs() < 0.3, "att={}", effect.ate);
        assert!(effect.se_bootstrap.is_some());
    }

    #[test]
    fn matching_index_reused_across_compatible_point_fits() {
        let (data, estimand) = confounded_scm(400, 7);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::Treated);
        let est = PropensityMatching { bootstrap_replicates: 0, ..PropensityMatching::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = PropensityEstimationWorkspace::default();
        let _ = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        let builds_after_first = ws.matching_index_builds;
        assert!(builds_after_first >= 1);
        let _ = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert_eq!(
            ws.matching_index_builds, builds_after_first,
            "identical donor geometry must not rebuild MatchingIndex"
        );
    }

    #[test]
    fn bootstrap_reuses_propensity_workspace_buffers() {
        let (data, estimand) = confounded_scm(400, 10);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = PropensityWeighting { bootstrap_replicates: 40, ..PropensityWeighting::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = PropensityEstimationWorkspace::default();
        let _ = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        let ols_grows = ws.propensity.ols.grow_count;
        let score_grows = ws.propensity.scores_grow_count;
        let scratch_ptr = ws.propensity.ols.scratch.as_ptr();
        let _ = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert_eq!(ws.propensity.ols.grow_count, ols_grows);
        assert_eq!(ws.propensity.scores_grow_count, score_grows);
        assert_eq!(ws.propensity.ols.scratch.as_ptr(), scratch_ptr);
    }

    #[test]
    fn propensity_matching_rejects_explicit_override() {
        let (data, estimand) = confounded_scm(200, 7);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::Treated);
        let est = PropensityMatching {
            overlap: OverlapPolicy::ExplicitOverride,
            ..PropensityMatching::new()
        };
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::Overlap { .. }));
    }

    #[test]
    fn distance_matching_recovers_att() {
        let (data, estimand) = confounded_scm(800, 8);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::Treated);
        let est = DistanceMatching { bootstrap_replicates: 30, ..DistanceMatching::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = PropensityEstimationWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!((effect.ate - 2.0).abs() < 0.3, "att={}", effect.ate);
        assert!(effect.se_bootstrap.is_some());
        assert!(effect.overlap_report.is_some());
    }

    #[test]
    fn distance_matching_rejects_explicit_override() {
        let (data, estimand) = confounded_scm(200, 9);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::Treated);
        let est = DistanceMatching {
            overlap: OverlapPolicy::ExplicitOverride,
            ..DistanceMatching::new()
        };
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::Overlap { .. }));
    }

    #[test]
    fn prepare_rejects_non_binary_treatment_column() {
        // {1,2}-coded treatment must be refused, not silently dichotomized at t > 0.5.
        let (t, y, z) = confounded_columns(100, 11);
        let t: Vec<f64> = t.iter().map(|&ti| ti + 1.0).collect();
        let (data, estimand) = build_dataset(t, y, z);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = PropensityWeighting::new();
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::Data(_)), "err={err:?}");
        assert!(err.to_string().contains("binary treatment column"), "err={err}");
    }

    #[test]
    fn hajek_difference_errors_on_zero_weight_arm() {
        // All treated weight trimmed away: must surface an error, not a silent NaN.
        let treatment = [1.0, 1.0, 0.0, 0.0];
        let outcome = [3.0, 4.0, 1.0, 2.0];
        let weights = [0.0, 0.0, 1.0, 1.0];
        let err = hajek_difference(&treatment, &outcome, &weights).unwrap_err();
        assert!(matches!(err, EstimationError::Data(_)), "err={err:?}");
    }

    #[test]
    fn stratification_trim_excludes_extreme_propensity_unit() {
        let (data, estimand) = confounded_scm_with_outlier(800, 12);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let untrimmed =
            PropensityStratification { bootstrap_replicates: 0, ..PropensityStratification::new() };
        let trimmed = PropensityStratification { overlap: trim_overlap(), ..untrimmed.clone() };

        let mut ws = PropensityEstimationWorkspace::default();
        let prep = untrimmed.prepare(&data, &estimand, &query).unwrap();
        let raw = untrimmed.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        let prep = trimmed.prepare(&data, &estimand, &query).unwrap();
        let clean = trimmed.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();

        assert!((raw.ate - 2.0).abs() > 1.0, "outlier should distort untrimmed ate={}", raw.ate);
        assert!((clean.ate - 2.0).abs() < 0.35, "trimmed ate={}", clean.ate);
        let report = clean.overlap_report.as_ref().unwrap();
        assert!(report.excluded_fraction > 0.0, "trim must report exclusions");
    }

    #[test]
    fn propensity_matching_trim_excludes_extreme_propensity_unit() {
        let (data, estimand) = confounded_scm_with_outlier(800, 13);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::Treated);
        let untrimmed = PropensityMatching { bootstrap_replicates: 0, ..PropensityMatching::new() };
        let trimmed = PropensityMatching { overlap: trim_overlap(), ..untrimmed.clone() };

        let mut ws = PropensityEstimationWorkspace::default();
        let prep = untrimmed.prepare(&data, &estimand, &query).unwrap();
        let raw = untrimmed.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        let prep = trimmed.prepare(&data, &estimand, &query).unwrap();
        let clean = trimmed.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();

        assert!((raw.ate - 2.0).abs() > 1.0, "outlier should distort untrimmed att={}", raw.ate);
        assert!((clean.ate - 2.0).abs() < 0.35, "trimmed att={}", clean.ate);
        let report = clean.overlap_report.as_ref().unwrap();
        assert!(report.excluded_fraction > 0.0, "trim must report exclusions");
    }

    #[test]
    fn distance_matching_trim_excludes_extreme_propensity_unit() {
        let (data, estimand) = confounded_scm_with_outlier(800, 14);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::Treated);
        let untrimmed = DistanceMatching { bootstrap_replicates: 0, ..DistanceMatching::new() };
        let trimmed = DistanceMatching { overlap: trim_overlap(), ..untrimmed.clone() };

        let mut ws = PropensityEstimationWorkspace::default();
        let prep = untrimmed.prepare(&data, &estimand, &query).unwrap();
        let raw = untrimmed.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        let prep = trimmed.prepare(&data, &estimand, &query).unwrap();
        let clean = trimmed.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();

        assert!((raw.ate - 2.0).abs() > 1.0, "outlier should distort untrimmed att={}", raw.ate);
        assert!((clean.ate - 2.0).abs() < 0.35, "trimmed att={}", clean.ate);
        let report = clean.overlap_report.as_ref().unwrap();
        assert!(report.excluded_fraction > 0.0, "trim must report exclusions");
    }
}
