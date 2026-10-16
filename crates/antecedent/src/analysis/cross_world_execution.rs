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
//! # Whole-path uncertainty
//!
//! [`CrossWorldExecution::whole_path_uncertainty`] derives, from the executed
//! row, which uncertainty sources the abduction-action-prediction path has, which
//! it propagates and how: the structural premise (declared, not propagated), the
//! exogenous posterior (exact by enumeration; the predictive variance of each
//! unit's quantity under it), the sampling of units (a declared
//! [`UnitSampling`], refused when undeclared) and the enumeration arithmetic.
//! The two spreads are for two estimands and are never added. The account is a
//! sealed sibling of the execution artifact, not a field of its wire: it is a
//! pure function of the execution's request and result and the declared unit
//! sampling, so a fresh consumer consumes the execution artifact and derives it
//! again ([`WholePathAccount::verify_against`]), refusing a stored account that
//! does not match.
//!
//! **Refresh.** New factual evidence is new units: the execution is sealed
//! again, and the account is **recomputed** (every unit's spread, the standard
//! errors and the identity); the declared [`UnitSampling`] is **kept** by being
//! passed again; an account is **refused** for a sampled or transported request
//! and for units with shared evidence declared independent.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use super::mechanism_exports::counterfactual_crate::selective::whole_path_uncertainty::{
    SourceAccount, SourceTreatment, UncertaintySource, UnitSampling, UnitSpread,
    WHOLE_PATH_UNCERTAINTY_VERSION, WholePathUncertainty, derive_whole_path_uncertainty,
};
pub use super::mechanism_exports::counterfactual_crate::selective::{
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
            "an unqualified interval or posterior of a declared model is closed: the \
             point value from exact enumeration is the result. The interval of a named unit \
             sampling is read with measured_interval, licensed only by a currently attesting \
             whole-path coverage record; a posterior statement has no parameter uncertainty \
             to describe and stays closed (docs/cross-world-execution.md)"
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

/// What an inspection of a whole-path account shows.
#[derive(Clone, Debug, PartialEq)]
pub struct WholePathInspection {
    /// Always `uncalibrated_spread`: no interval or posterior reading is licensed.
    pub label: &'static str,
    /// The declared unit sampling.
    pub unit_sampling: &'static str,
    /// Mean of the unit values.
    pub value: f64,
    /// `(source, treatment)` in the fixed source order.
    pub sources: Vec<(&'static str, String)>,
    /// Standard error of the realized-outcome mean about the mean of the unit
    /// probabilities.
    pub realized_mean_standard_error: f64,
    /// Standard error of the mean unit probability as a population-mean
    /// estimate, when determined.
    pub unit_mean_standard_error: Option<f64>,
    /// Identity of the account.
    pub identity: String,
}

/// The whole-path uncertainty of an executed cross-world row, bound to the
/// execution it was derived from.
#[derive(Clone, Debug, PartialEq)]
pub struct WholePathAccount {
    uncertainty: WholePathUncertainty,
    execution_receipt_digest: String,
}

impl WholePathAccount {
    /// The derived account.
    #[must_use]
    pub const fn uncertainty(&self) -> &WholePathUncertainty {
        &self.uncertainty
    }

    /// The receipt digest of the execution the account stands on.
    #[must_use]
    pub fn execution_receipt_digest(&self) -> &str {
        &self.execution_receipt_digest
    }

    /// Derive the account again from `execution` and require it to equal this
    /// one: a stored account of other units, mechanisms, quantity or unit
    /// sampling is refused, never annotated.
    ///
    /// # Errors
    /// Every refusal of [`CrossWorldExecution::whole_path_uncertainty`] and a
    /// `cross_world_uncertainty.account_mismatch` refusal for an account that
    /// differs from the derivation.
    pub fn verify_against(
        &self,
        execution: &CrossWorldExecution,
        ctx: &ExecutionContext,
    ) -> Result<(), CrossWorldExecutionError> {
        let fresh = execution.whole_path_uncertainty(self.uncertainty.unit_sampling, ctx)?;
        if fresh != *self {
            return Err(CrossWorldExecutionError::Refused(Box::new(CrossWorldRefusal {
                refusal: antecedent_contract::StructuredRefusal::new(
                    antecedent_core::reason_code!("invalid_argument"),
                    "cross_world_uncertainty",
                    "cross_world_uncertainty.account_mismatch",
                )
                .with_remedy(
                    "derive the account from the consumed execution; a stored account is \
                     never trusted",
                ),
                missing: Vec::new(),
            })));
        }
        Ok(())
    }

    /// Plain-data summary for inspection.
    #[must_use]
    pub fn inspect(&self) -> WholePathInspection {
        let u = &self.uncertainty;
        WholePathInspection {
            label: u.label,
            unit_sampling: u.unit_sampling.name(),
            value: u.value,
            sources: u
                .sources
                .iter()
                .map(|a| (a.source.name(), format!("{:?}", a.treatment)))
                .collect(),
            realized_mean_standard_error: u.realized_mean_standard_error(),
            unit_mean_standard_error: u.unit_mean_standard_error(),
            identity: u.identity.clone(),
        }
    }
}

impl CrossWorldExecution {
    /// Derive the whole-path uncertainty of this execution under a declared unit
    /// sampling.
    ///
    /// The derivation re-abduces and re-predicts every unit from the sealed
    /// request and requires every unit value and the mean to equal the executed
    /// result bit for bit. Every spread carries the label `uncalibrated_spread`;
    /// [`Self::interval`] and [`Self::posterior`] stay closed.
    ///
    /// # Errors
    /// Undeclared unit sampling, a sampled or transported request, a coupling the
    /// joint-law rule does not license, units with shared evidence declared
    /// independent, an exceeded bound, or cancellation.
    pub fn whole_path_uncertainty(
        &self,
        unit_sampling: UnitSampling,
        ctx: &ExecutionContext,
    ) -> Result<WholePathAccount, CrossWorldExecutionError> {
        let uncertainty = derive_whole_path_uncertainty(
            self.artifact.request(),
            self.artifact.result(),
            unit_sampling,
            ctx,
        )?;
        Ok(WholePathAccount {
            uncertainty,
            execution_receipt_digest: self.artifact.identity().receipt_digest.clone(),
        })
    }
}
