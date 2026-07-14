//! Validator contract (DESIGN.md §18.1).
//!
//! Effect refuters (§18.2) implement this trait. The trait is named `Validator`
//! (not `Refuter`) per DESIGN.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

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

/// Validation / refutation algorithm over artifact type `A` (DESIGN §18.1).
///
/// `validate` also takes an [`EstimationWorkspace`] because effect refuters in
/// this crate refit estimators; DESIGN's sketch omits it.
pub trait Validator<A> {
    /// Prepared artifact produced by [`Self::prepare`].
    type Prepared;
    /// Report produced by [`Self::validate`].
    type Report;

    /// Compile `artifact` into a reusable prepared form.
    ///
    /// # Errors
    ///
    /// Incompatible artifact or missing prerequisites.
    fn prepare(
        &self,
        artifact: &A,
        ctx: &ExecutionContext,
    ) -> Result<Self::Prepared, ValidationError>;

    /// Run the check on a prepared artifact.
    ///
    /// # Errors
    ///
    /// Data, estimation, or applicability failures.
    fn validate(
        &self,
        prepared: &mut Self::Prepared,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<Self::Report, ValidationError>;
}

/// Prepared effect-refutation problem (borrowed inputs + no extra state).
#[derive(Clone, Copy, Debug)]
pub struct PreparedRefutation<'a> {
    /// Underlying refutation problem.
    pub problem: RefutationProblem<'a>,
}

/// Run `validator` end-to-end (prepare → validate) for suite dispatch.
///
/// # Errors
///
/// Propagates prepare/validate failures.
pub fn run_validator<'a, V>(
    validator: &V,
    problem: &RefutationProblem<'a>,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
) -> Result<RefutationReport, ValidationError>
where
    V: Validator<RefutationProblem<'a>, Prepared = PreparedRefutation<'a>, Report = RefutationReport>,
{
    let mut prepared = validator.prepare(problem, ctx)?;
    validator.validate(&mut prepared, workspace, ctx)
}

macro_rules! impl_effect_validator {
    ($ty:ty, $call:expr) => {
        impl<'a> Validator<RefutationProblem<'a>> for $ty {
            type Prepared = PreparedRefutation<'a>;
            type Report = RefutationReport;

            fn prepare(
                &self,
                artifact: &RefutationProblem<'a>,
                _ctx: &ExecutionContext,
            ) -> Result<Self::Prepared, ValidationError> {
                Ok(PreparedRefutation { problem: *artifact })
            }

            fn validate(
                &self,
                prepared: &mut Self::Prepared,
                workspace: &mut EstimationWorkspace,
                ctx: &ExecutionContext,
            ) -> Result<Self::Report, ValidationError> {
                ($call)(self, &prepared.problem, workspace, ctx)
            }
        }
    };
}

impl_effect_validator!(PlaceboTreatment, |this: &PlaceboTreatment, p, ws, ctx| {
    this.refute(p, ws, ctx)
});
impl_effect_validator!(RandomCommonCause, |this: &RandomCommonCause, p, ws, ctx| {
    this.refute(p, ws, ctx)
});
impl_effect_validator!(BootstrapRefute, |this: &BootstrapRefute, p, ws, ctx| {
    this.refute(p, ws, ctx)
});
impl_effect_validator!(UnobservedCommonCause, |this: &UnobservedCommonCause, p, ws, ctx| {
    this.refute(p, ws, ctx)
});
impl_effect_validator!(DataSubsetRefuter, |this: &DataSubsetRefuter, p, ws, ctx| {
    this.refute(p, ws, ctx)
});
impl_effect_validator!(DummyOutcome, |this: &DummyOutcome, p, ws, ctx| {
    this.refute(p, ws, ctx)
});
impl_effect_validator!(GraphRefuter, |this: &GraphRefuter, p, ws, ctx| {
    this.refute(p, ws, ctx)
});
impl_effect_validator!(LinearSensitivity, |this: &LinearSensitivity, p, ws, ctx| {
    this.refute(p, ws, ctx)
});
impl_effect_validator!(PartialLinearSensitivity, |this: &PartialLinearSensitivity, p, ws, ctx| {
    this.refute(p, ws, ctx)
});
impl_effect_validator!(NonparametricSensitivity, |this: &NonparametricSensitivity, p, ws, ctx| {
    this.refute(p, ws, ctx)
});
impl_effect_validator!(ReiszSensitivity, |this: &ReiszSensitivity, p, ws, ctx| {
    this.refute(p, ws, ctx)
});

/// Validators whose `refute` does not need an estimation workspace.
macro_rules! impl_stateless_effect_validator {
    ($ty:ty, $call:expr) => {
        impl<'a> Validator<RefutationProblem<'a>> for $ty {
            type Prepared = PreparedRefutation<'a>;
            type Report = RefutationReport;

            fn prepare(
                &self,
                artifact: &RefutationProblem<'a>,
                _ctx: &ExecutionContext,
            ) -> Result<Self::Prepared, ValidationError> {
                Ok(PreparedRefutation { problem: *artifact })
            }

            fn validate(
                &self,
                prepared: &mut Self::Prepared,
                _workspace: &mut EstimationWorkspace,
                _ctx: &ExecutionContext,
            ) -> Result<Self::Report, ValidationError> {
                ($call)(self, &prepared.problem)
            }
        }
    };
}

impl_stateless_effect_validator!(OverlapRefuter, |this: &OverlapRefuter, p| this.refute(p));
impl_stateless_effect_validator!(OverlapRuleRefuter, |this: &OverlapRuleRefuter, p| {
    this.refute(p)
});
impl_stateless_effect_validator!(EValue, |this: &EValue, p| this.refute(p));
