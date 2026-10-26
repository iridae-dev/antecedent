//! The execution settings every checked temporal graph route freezes beside its
//! proof payload: the selected procedure, inference mode, bootstrap budget,
//! validation suite, completion cap and the sealed physical plan.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::CausalQuery;

use crate::InferenceMode;
use crate::error::CausalError;
use crate::planner::PhysicalExecutionPlan;
use crate::strategy_table::{EstimatorId, IdentifierId};

use super::builder::RefuteSuite;
use super::execute::Study;

/// How much of the sealed procedure the physical plan must name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PlanBindingScope {
    /// The plan names exactly the frozen identifier and estimator.
    Procedure,
    /// The plan names exactly the frozen estimator; the identifier is not
    /// constrained by the plan record.
    Estimator,
}

/// Frozen procedure settings shared by the checked temporal graph operations.
/// Each operation composes one of these with its proof payload and its own
/// query, so the binding guard and the accessors exist once.
#[derive(Clone, Debug)]
pub(crate) struct CheckedRouteFrozen {
    identifier: IdentifierId,
    estimator: EstimatorId,
    inference: InferenceMode,
    bootstrap_replicates: u32,
    validation: RefuteSuite,
    physical: Arc<PhysicalExecutionPlan>,
    max_completions: Option<usize>,
}

impl CheckedRouteFrozen {
    pub(crate) fn new(
        identifier: IdentifierId,
        estimator: EstimatorId,
        inference: InferenceMode,
        bootstrap_replicates: u32,
        validation: RefuteSuite,
        physical: Arc<PhysicalExecutionPlan>,
        max_completions: Option<usize>,
    ) -> Self {
        Self {
            identifier,
            estimator,
            inference,
            bootstrap_replicates,
            validation,
            physical,
            max_completions,
        }
    }

    /// Require the sealed plan to carry exactly `query` and the frozen
    /// procedure (as far as `scope` constrains it).
    ///
    /// # Errors
    ///
    /// A compile error carrying `message` when the plan disagrees.
    pub(crate) fn validate_plan_binding(
        &self,
        query: &CausalQuery,
        scope: PlanBindingScope,
        message: &str,
    ) -> Result<(), CausalError> {
        let binding = self.physical.binding();
        let procedure_matches = match scope {
            PlanBindingScope::Procedure => binding.names(self.identifier, self.estimator),
            PlanBindingScope::Estimator => binding.estimator.is(self.estimator),
        };
        if &self.physical.logical.query != query || !procedure_matches {
            return Err(CausalError::Compile { message: message.into() });
        }
        Ok(())
    }

    /// Bind every frozen procedure setting onto a click study, so the live
    /// builder cannot change the procedure after preparation. Caller
    /// validators never run on a frozen route.
    pub(crate) fn bind_study(&self, study: &mut Study) {
        study.inference = self.inference.clone();
        study.identifier = Some(self.identifier);
        study.estimator = Some(self.estimator);
        study.bootstrap_replicates = self.bootstrap_replicates;
        study.refute = self.validation;
        study.custom_validators.clear();
        study.max_completions = self.max_completions;
    }

    /// Add the identifier, estimator, bootstrap budget and validation suite to
    /// an operation's `Debug` output.
    pub(crate) fn debug_procedure(&self, debug: &mut std::fmt::DebugStruct<'_, '_>) {
        debug
            .field("identifier", &self.identifier)
            .field("estimator", &self.estimator)
            .field("bootstrap_replicates", &self.bootstrap_replicates)
            .field("validation", &self.validation);
    }

    pub(crate) const fn identifier(&self) -> IdentifierId {
        self.identifier
    }

    pub(crate) const fn estimator(&self) -> EstimatorId {
        self.estimator
    }

    pub(crate) const fn inference(&self) -> &InferenceMode {
        &self.inference
    }

    pub(crate) const fn bootstrap_replicates(&self) -> u32 {
        self.bootstrap_replicates
    }

    pub(crate) const fn validation(&self) -> RefuteSuite {
        self.validation
    }

    pub(crate) fn physical(&self) -> &PhysicalExecutionPlan {
        &self.physical
    }

    pub(crate) const fn max_completions(&self) -> Option<usize> {
        self.max_completions
    }
}
