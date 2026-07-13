//! Validation suite orchestration (DESIGN.md §18.5).
//!
//! Runs requested validators, returning explicit [`ValidationOutcome::NotApplicable`] when a
//! check is incompatible with the estimator/estimand (rather than failing the whole suite).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::many_single_char_names, clippy::unused_self)]

use std::sync::Arc;

use causal_core::ExecutionContext;
use causal_estimate::EstimationWorkspace;

use crate::bootstrap_refute::BootstrapRefute;
use crate::common::{RefutationProblem, RefutationReport};
use crate::data_subset::DataSubsetRefuter;
use crate::dummy_outcome::DummyOutcome;
use crate::error::ValidationError;
use crate::evalue::EValue;
use crate::graph_refute::GraphRefuter;
use crate::overlap::OverlapRefuter;
use crate::overlap_rule::OverlapRuleRefuter;
use crate::placebo::PlaceboTreatment;
use crate::rcc::RandomCommonCause;
use crate::reisz::ReiszSensitivity;
use crate::sensitivity::{LinearSensitivity, NonparametricSensitivity, PartialLinearSensitivity};
use crate::unobserved_common_cause::UnobservedCommonCause;

/// Named validators that can be attached to a [`ValidationSuite`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ValidatorId {
    /// Placebo treatment.
    Placebo,
    /// Random common cause.
    RandomCommonCause,
    /// Bootstrap refutation.
    Bootstrap,
    /// Unobserved common cause.
    UnobservedCommonCause,
    /// Overlap / positivity assessment.
    Overlap,
    /// Overlap-rule / trimming assessment.
    OverlapRule,
    /// Data subset.
    DataSubset,
    /// Dummy outcome.
    DummyOutcome,
    /// E-value.
    EValue,
    /// Graph refutation (drop adjustment members).
    Graph,
    /// Linear sensitivity.
    LinearSensitivity,
    /// Partial-linear sensitivity.
    PartialLinearSensitivity,
    /// Nonparametric sensitivity.
    NonparametricSensitivity,
    /// Reisz-representer sensitivity.
    Reisz,
}

/// Outcome of one validator in a suite.
#[derive(Clone, Debug)]
pub enum ValidationOutcome {
    /// Validator ran and produced a report.
    Report(RefutationReport),
    /// Validator was requested but is incompatible with this problem.
    NotApplicable {
        /// Validator id.
        validator: ValidatorId,
        /// Why it was skipped.
        reason: Arc<str>,
    },
}

/// Ordered suite of validators (DESIGN §18.5).
#[derive(Clone, Debug, Default)]
pub struct ValidationSuite {
    validators: Vec<ValidatorId>,
}

impl ValidationSuite {
    /// Empty suite.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a validator (order preserved).
    #[must_use]
    pub fn with(mut self, id: ValidatorId) -> Self {
        self.validators.push(id);
        self
    }

    /// Placebo + RCC (legacy default).
    #[must_use]
    pub fn placebo_and_rcc() -> Self {
        Self::new().with(ValidatorId::Placebo).with(ValidatorId::RandomCommonCause)
    }

    /// Full Phase 4 effect-validation set.
    #[must_use]
    pub fn full_effect() -> Self {
        Self::new()
            .with(ValidatorId::Placebo)
            .with(ValidatorId::RandomCommonCause)
            .with(ValidatorId::Bootstrap)
            .with(ValidatorId::UnobservedCommonCause)
            .with(ValidatorId::Overlap)
            .with(ValidatorId::OverlapRule)
            .with(ValidatorId::DataSubset)
            .with(ValidatorId::DummyOutcome)
            .with(ValidatorId::EValue)
            .with(ValidatorId::Graph)
            .with(ValidatorId::LinearSensitivity)
            .with(ValidatorId::PartialLinearSensitivity)
            .with(ValidatorId::NonparametricSensitivity)
            .with(ValidatorId::Reisz)
    }

    /// Run all configured validators.
    ///
    /// # Errors
    ///
    /// Propagates hard failures from applicable validators (not `NotApplicable` skips).
    pub fn run(
        &self,
        problem: &RefutationProblem<'_>,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<Vec<ValidationOutcome>, ValidationError> {
        let mut out = Vec::with_capacity(self.validators.len());
        for &id in &self.validators {
            out.push(self.run_one(id, problem, workspace, ctx)?);
        }
        Ok(out)
    }

    /// Collect only successful [`RefutationReport`]s (drops `NotApplicable`).
    #[must_use]
    pub fn reports_only(outcomes: &[ValidationOutcome]) -> Vec<RefutationReport> {
        outcomes
            .iter()
            .filter_map(|o| match o {
                ValidationOutcome::Report(r) => Some(r.clone()),
                ValidationOutcome::NotApplicable { .. } => None,
            })
            .collect()
    }

    fn run_one(
        &self,
        id: ValidatorId,
        problem: &RefutationProblem<'_>,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<ValidationOutcome, ValidationError> {
        let linear_ok = &*problem.estimand.method == "backdoor.adjustment"
            && problem.estimator.is_none_or(|e| e == "linear.adjustment.ate");
        match id {
            ValidatorId::Placebo => {
                if !linear_ok {
                    return Ok(na(
                        id,
                        "PlaceboTreatment requires backdoor.adjustment + linear path",
                    ));
                }
                Ok(ValidationOutcome::Report(
                    PlaceboTreatment::new().refute(problem, workspace, ctx)?,
                ))
            }
            ValidatorId::RandomCommonCause => {
                if !linear_ok {
                    return Ok(na(
                        id,
                        "RandomCommonCause requires backdoor.adjustment + linear path",
                    ));
                }
                Ok(ValidationOutcome::Report(
                    RandomCommonCause::new().refute(problem, workspace, ctx)?,
                ))
            }
            ValidatorId::Bootstrap => {
                if !linear_ok {
                    return Ok(na(
                        id,
                        "BootstrapRefute requires backdoor.adjustment + linear path",
                    ));
                }
                Ok(ValidationOutcome::Report(
                    BootstrapRefute::new().refute(problem, workspace, ctx)?,
                ))
            }
            ValidatorId::UnobservedCommonCause => {
                if !linear_ok {
                    return Ok(na(id, "UnobservedCommonCause requires backdoor.adjustment"));
                }
                Ok(ValidationOutcome::Report(
                    UnobservedCommonCause::new().refute(problem, workspace, ctx)?,
                ))
            }
            ValidatorId::Overlap => {
                Ok(ValidationOutcome::Report(OverlapRefuter::new().refute(problem)?))
            }
            ValidatorId::OverlapRule => {
                Ok(ValidationOutcome::Report(OverlapRuleRefuter::new().refute(problem)?))
            }
            ValidatorId::DataSubset => {
                if !linear_ok {
                    return Ok(na(
                        id,
                        "DataSubsetRefuter requires backdoor.adjustment + linear path",
                    ));
                }
                Ok(ValidationOutcome::Report(
                    DataSubsetRefuter::new().refute(problem, workspace, ctx)?,
                ))
            }
            ValidatorId::DummyOutcome => {
                if !linear_ok {
                    return Ok(na(id, "DummyOutcome requires backdoor.adjustment + linear path"));
                }
                Ok(ValidationOutcome::Report(DummyOutcome::new().refute(problem, workspace, ctx)?))
            }
            ValidatorId::EValue => Ok(ValidationOutcome::Report(EValue::new().refute(problem)?)),
            ValidatorId::Graph => {
                if !linear_ok {
                    return Ok(na(id, "GraphRefuter requires backdoor.adjustment + linear path"));
                }
                Ok(ValidationOutcome::Report(GraphRefuter::new().refute(problem, workspace, ctx)?))
            }
            ValidatorId::LinearSensitivity => {
                if !linear_ok {
                    return Ok(na(id, "LinearSensitivity requires backdoor.adjustment"));
                }
                Ok(ValidationOutcome::Report(
                    LinearSensitivity::new().refute(problem, workspace, ctx)?,
                ))
            }
            ValidatorId::PartialLinearSensitivity => {
                if !linear_ok {
                    return Ok(na(id, "PartialLinearSensitivity requires backdoor.adjustment"));
                }
                Ok(ValidationOutcome::Report(
                    PartialLinearSensitivity::new().refute(problem, workspace, ctx)?,
                ))
            }
            ValidatorId::NonparametricSensitivity => Ok(ValidationOutcome::Report(
                NonparametricSensitivity::new().refute(problem, workspace, ctx)?,
            )),
            ValidatorId::Reisz => Ok(ValidationOutcome::Report(
                ReiszSensitivity::new().refute(problem, workspace, ctx)?,
            )),
        }
    }
}

fn na(id: ValidatorId, reason: &str) -> ValidationOutcome {
    ValidationOutcome::NotApplicable { validator: id, reason: Arc::from(reason) }
}

#[cfg(test)]
mod tests {
    use causal_core::{
        AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
        RoleHint, SmallRoleSet, ValueType, VariableId,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use causal_estimate::{EstimationWorkspace, LinearAdjustmentAte};
    use causal_expr::ExprId;
    use causal_identify::IdentifiedEstimand;

    use super::*;
    use crate::common::RefutationProblem;

    fn toy() -> (TabularData, IdentifiedEstimand) {
        let n = 120usize;
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
    fn full_suite_runs_applicable_validators() {
        let (data, estimand) = toy();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        let ctx = ExecutionContext::for_tests(2);
        let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        let problem = RefutationProblem {
            data: &data,
            estimand: &estimand,
            query: &query,
            original: &original,
            estimator: Some("linear.adjustment.ate"),
        };
        let outcomes = ValidationSuite::full_effect().run(&problem, &mut ws, &ctx).unwrap();
        assert_eq!(outcomes.len(), 14);
        let reports = ValidationSuite::reports_only(&outcomes);
        assert!(reports.len() >= 10, "reports={}", reports.len());
    }
}
