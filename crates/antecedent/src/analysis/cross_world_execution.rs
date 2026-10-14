//! General cross-world selective execution through the analysis facade.
//!
//! One staged pipeline (mechanism fitting, factual abduction, exogenous draws,
//! action-specific prediction, summary) executes a static Markovian SCM over
//! finite variables and, as its fixed-population temporal row, the two-slice
//! additive linear SCM. Which quantities are executed, which keep their own
//! theorem-scoped route and which are closed is the checked capability matrix
//! ([`capability_matrix`], [`decide_capability`]); a request outside it is a
//! scientific refusal that names the unmet prerequisite, never a cache miss.
//!
//! * **Execute / seal** ([`CrossWorldExecution::seal`]) enumerates each unit's
//!   exogenous posterior exactly and evaluates every named world under the
//!   same posterior states, so natural and path-specific contrasts and joint
//!   events are computed from coupled worlds only. Independent per-world noise
//!   is never paired as a counterfactual joint law: that decision is the one
//!   joint-law rule, `antecedent_contract::check_propagation`.
//! * **Reuse** ([`FiniteScmExecutor`]) keeps each unit's abduction under its
//!   canonical dependency identity: an action-only change reuses it, a changed
//!   evidence, mechanism parameter, structure, population or history rebuilds
//!   it. Observed work counters name what ran.
//! * **Export / consume** seal the declarations, the shared exogenous states
//!   and the numbers into `cross_world_execution_v1`; consume replays them with
//!   a fresh executor and returns it warm for a licensed resume.
//!
//! # What a value means
//!
//! A value is a `point_only` result of exact enumeration under a declared
//! model. Mechanism adequacy is a named premise, not a check. Interval and
//! posterior readings are closed ([`CrossWorldExecution::interval`] and
//! [`CrossWorldExecution::posterior`] refuse): the whole abduction-action-
//! prediction path has no measured coverage record, and the calibration design
//! that would open it is recorded in `docs/cross-world-execution.md`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_counterfactual::selective::{
    CapabilityStatus, ConditioningKind, CouplingKind, CrossWorldExecutionError, CrossWorldRefusal,
    EvidenceKind, ExecutionReceipt, ExecutionStage, ExogenousInference, ExogenousVariable,
    FactualUnit, FiniteExecutionRequest, FiniteExecutionResult, FiniteQuantity, FiniteScm,
    FiniteScmExecutor, FiniteVariable, FiniteWorld, HistoryKind, InterventionKind, MatrixDecision,
    MatrixRefusal, MatrixRequest, MatrixRow, ModelClass, NoiseCoupling, PopulationKind,
    QuantityClass, RetainedAbduction, StageReceipt, StageStatus, TemporalExecution,
    TemporalExecutor, UnitAnswer, WorkCounters, WorldAtom, WorldRoute, decide as decide_capability,
    rows as capability_matrix, stream_identity,
};
pub use antecedent_io::cross_world_execution_artifact::{
    CROSS_WORLD_EXECUTION_ARTIFACT_VERSION, CrossWorldExecutionArtifact,
    CrossWorldExecutionArtifactError, CrossWorldExecutionIdentity, CrossWorldExecutionMeta,
    CrossWorldExecutionRefusalWire,
};

use antecedent_core::ExecutionContext;

/// A sealed or consumed cross-world execution.
#[derive(Debug)]
pub struct CrossWorldExecution {
    artifact: CrossWorldExecutionArtifact,
}

fn closed_reading(reading: &'static str, detail: &'static str) -> CrossWorldExecutionRefusalWire {
    CrossWorldExecutionRefusalWire {
        code: antecedent_core::reason_code!("cell_not_licensed").to_owned(),
        stage: "execute".to_owned(),
        detail: detail.to_owned(),
        offending: Some(reading.to_owned()),
        remedy: Some(
            "the whole abduction-action-prediction path has no measured coverage record; the \
             point value from exact enumeration is the result, and the calibration design that \
             would open this reading is recorded in docs/cross-world-execution.md"
                .to_owned(),
        ),
        missing: vec!["whole_path_calibration_record".to_owned()],
    }
}

impl CrossWorldExecution {
    /// Execute `request` with a fresh executor and seal the answer.
    ///
    /// # Errors
    /// A typed refusal (matrix, coupling, readings, population, history,
    /// refuted evidence), cancellation or a format bound.
    pub fn seal(
        request: &FiniteExecutionRequest,
        ctx: &ExecutionContext,
    ) -> Result<Self, CrossWorldExecutionArtifactError> {
        CrossWorldExecutionArtifact::seal(request, ctx).map(|artifact| Self { artifact })
    }

    /// Consume exported bytes by replaying them with a fresh executor.
    ///
    /// # Errors
    /// Corruption, another version, a changed identity, numbers or states that
    /// do not replay, a refusal or cancellation.
    pub fn consume(
        bytes: &[u8],
        expected: Option<&CrossWorldExecutionIdentity>,
        ctx: &ExecutionContext,
    ) -> Result<Self, CrossWorldExecutionArtifactError> {
        CrossWorldExecutionArtifact::from_bytes(bytes, expected, ctx)
            .map(|artifact| Self { artifact })
    }

    /// The answer.
    #[must_use]
    pub fn result(&self) -> &FiniteExecutionResult {
        self.artifact.result()
    }

    /// The identity digests of the declarations and stages.
    #[must_use]
    pub fn identity(&self) -> &CrossWorldExecutionIdentity {
        self.artifact.identity()
    }

    /// The sealed artifact.
    #[must_use]
    pub fn artifact(&self) -> &CrossWorldExecutionArtifact {
        &self.artifact
    }

    /// Serialize through the checksummed container.
    ///
    /// # Errors
    /// An empty id, an encoding failure or an oversized payload.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, CrossWorldExecutionArtifactError> {
        self.artifact.to_bytes(artifact_id)
    }

    /// Run a further request through the verified executor: an action-only
    /// change reuses the abduction this execution verified.
    ///
    /// # Errors
    /// As [`FiniteScmExecutor::execute`].
    pub fn resume(
        &mut self,
        request: &FiniteExecutionRequest,
        ctx: &ExecutionContext,
    ) -> Result<FiniteExecutionResult, CrossWorldExecutionError> {
        self.artifact.executor_mut().execute(request, ctx)
    }

    #[allow(
        clippy::result_large_err,
        reason = "a refusal is the cold path of a once-per-call check; boxing the error would change the public error type"
    )]
    /// An interval for the quantity. Closed.
    ///
    /// # Errors
    /// Always: `cell_not_licensed` (`cross_world_execution.interval_route_closed`).
    pub fn interval(&self) -> Result<(f64, f64), CrossWorldExecutionRefusalWire> {
        Err(closed_reading("interval", "cross_world_execution.interval_route_closed"))
    }

    #[allow(
        clippy::result_large_err,
        reason = "a refusal is the cold path of a once-per-call check; boxing the error would change the public error type"
    )]
    /// A posterior statement for the quantity. Closed.
    ///
    /// # Errors
    /// Always: `cell_not_licensed` (`cross_world_execution.posterior_route_closed`).
    pub fn posterior(&self) -> Result<(f64, f64), CrossWorldExecutionRefusalWire> {
        Err(closed_reading("posterior", "cross_world_execution.posterior_route_closed"))
    }
}
