//! Inference from adaptively collected data: prepare the retained analysis
//! contract, run one inferential paradigm, export and independently consume.
//!
//! An adaptive experiment's data are not a fixed sample. This module routes a
//! request through the layers that own each rule and adds no scientific rule
//! of its own:
//!
//! 1. **Prepare** ([`AdaptiveInferenceSession::prepare`]) validates the
//!    retained analysis contract (`antecedent_contract::adaptive_design`):
//!    the assignment-policy history, the realized assignment probabilities,
//!    the exposure history, outcome availability and delay, the interim
//!    analyses and the stopping rule. A missing element is a typed refusal,
//!    one detail per element, and nothing is defaulted. The final sample is
//!    never analysed as if the design had been fixed.
//! 2. **Analyse** ([`AdaptiveInferenceSession::analyse`]) authorizes one
//!    inferential paradigm for the contract's procedure cell and runs it: a
//!    Bayesian posterior, a decision-theoretic stopping decision, adaptively
//!    weighted scores, an anytime-valid confidence sequence or group-sequential
//!    boundaries. Every result is point-only; the interval and guarantee
//!    routes refuse with `cell_not_licensed` until whole-procedure coverage of
//!    the exact procedure is measured.
//! 3. **Export** ([`AdaptiveInferenceSession::export_contract`] and
//!    [`AdaptiveInferenceSession::export_result`]) writes the monitoring
//!    contract (`adaptive_monitoring_contract_v1`) or a result bound to it
//!    (`adaptive_inference_result_v1`).
//! 4. **Consume** ([`consume_adaptive_monitoring_contract_artifact`] and
//!    [`consume_adaptive_inference_result_artifact`]) replays the contract and
//!    recomputes the stored values, refusing any edit.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_contract::adaptive_design::{
    AdaptiveDesignContract, AdaptiveDesignDeclaration, InferenceParadigm, InterimEvidence,
    ProcedureCell, StoppingRuleDeclaration, procedure_status,
};
pub use antecedent_contract::adaptive_design::{
    AssignmentFamily, BoundarySidedness, CellStatus, DelayMechanism, InterimAnalysis,
    MonitoringStrategy, OutcomeAvailability, OutcomeStructure, PolicyEpoch, SpendingKind,
    StoppingRuleKind, StoppingSpec, licensed_routes, procedure_cells,
};
use antecedent_core::ExecutionContext;
pub use antecedent_io::adaptive_inference_result_artifact::AdaptiveResultSpec as AdaptiveRequest;
use antecedent_io::adaptive_inference_result_artifact::{
    AdaptiveInferenceResultArtifact, AdaptiveResult, AdaptiveResultError, AdaptiveResultSpec,
    compute_adaptive_result,
};
use antecedent_io::adaptive_monitoring_contract_artifact::AdaptiveMonitoringContractArtifact;

/// One refusal as plain data: registered reason code, namespaced
/// `adaptive_inference.*` detail, and what would license the request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdaptiveInferenceRefusal {
    /// Registered runtime reason code.
    pub code: &'static str,
    /// Namespaced detail, `adaptive_inference.<slot>`.
    pub detail: String,
    /// What was refused and what would license it.
    pub message: String,
}

impl std::fmt::Display for AdaptiveInferenceRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}{}: {}: {}",
            antecedent_core::reason_code::PREFIX,
            self.code,
            self.detail,
            self.message
        )
    }
}

impl std::error::Error for AdaptiveInferenceRefusal {}

impl From<AdaptiveResultError> for AdaptiveInferenceRefusal {
    fn from(error: AdaptiveResultError) -> Self {
        Self { code: error.code, detail: error.detail, message: error.message }
    }
}

fn from_external(refusal: &antecedent_contract::ExternalRefusal) -> AdaptiveInferenceRefusal {
    AdaptiveInferenceRefusal {
        code: refusal.code,
        detail: refusal.detail.clone(),
        message: refusal.remedy.unwrap_or("").to_owned(),
    }
}

/// A result bound to the contract and request it was computed from.
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveInferenceOutcome {
    /// The request.
    pub request: AdaptiveResultSpec,
    /// The paradigm the result belongs to.
    pub paradigm: InferenceParadigm,
    /// The procedure the experiment ran under.
    pub procedure: ProcedureCell,
    /// The stopping rule the inference used.
    pub stopping_rule: Option<StoppingRuleDeclaration>,
    /// The computed values and their method.
    pub result: AdaptiveResult,
    /// Identity (hex) of the contract the result was computed under.
    pub contract_identity: String,
}

/// A prepared adaptive-experiment analysis.
#[derive(Clone, Debug)]
pub struct AdaptiveInferenceSession {
    contract: AdaptiveDesignContract,
}

impl AdaptiveInferenceSession {
    /// Validate the retained analysis contract.
    ///
    /// # Errors
    /// One refusal per missing element and per inconsistency.
    pub fn prepare(
        declaration: AdaptiveDesignDeclaration,
    ) -> Result<Self, Vec<AdaptiveInferenceRefusal>> {
        AdaptiveDesignContract::new(declaration)
            .map(|contract| Self { contract })
            .map_err(|error| error.refusals.iter().map(from_external).collect())
    }

    /// The validated contract.
    #[must_use]
    pub const fn contract(&self) -> &AdaptiveDesignContract {
        &self.contract
    }

    /// Identity (hex) of the contract.
    #[must_use]
    pub fn contract_identity(&self) -> String {
        self.contract.identity().to_hex()
    }

    /// What the procedure matrix does with this experiment's cell.
    #[must_use]
    pub fn cell_status(&self) -> CellStatus {
        procedure_status(&self.contract.cell())
    }

    /// The observations behind each interim analysis, counted once.
    ///
    /// # Errors
    /// A ledger bound exceeded.
    pub fn interim_evidence(&self) -> Result<InterimEvidence, AdaptiveInferenceRefusal> {
        self.contract.interim_evidence().map_err(|r| from_external(&r))
    }

    /// Analyse the data as if the design had been fixed in advance.
    ///
    /// # Errors
    /// Always for data from an adaptive design: the fixed-sample paradigm is
    /// refused with `adaptive_design_ignored`. Only a design that was fixed in
    /// advance (fixed randomization, one final analysis, no stopping rule) is
    /// authorized, and its fixed-sample analysis is the existing randomized
    /// route, not computed here.
    pub fn fixed_sample(&self) -> Result<(), AdaptiveInferenceRefusal> {
        self.contract
            .authorize(InferenceParadigm::FrequentistFixedSample)
            .map(|_| ())
            .map_err(|r| from_external(&r))
    }

    /// Run the inferential paradigm a request names.
    ///
    /// # Errors
    /// A paradigm the cell is not licensed for, an exposure or delay that
    /// fails the authorization, a probability floor failure, an outcome
    /// outside the route's range, or cancellation.
    pub fn analyse(
        &self,
        request: &AdaptiveResultSpec,
        ctx: &ExecutionContext,
    ) -> Result<AdaptiveInferenceOutcome, AdaptiveInferenceRefusal> {
        let result = compute_adaptive_result(&self.contract, request, ctx)?;
        Ok(AdaptiveInferenceOutcome {
            request: request.clone(),
            paradigm: request.paradigm(),
            procedure: self.contract.cell(),
            stopping_rule: self.contract.declaration().stopping_rule,
            result,
            contract_identity: self.contract_identity(),
        })
    }

    /// Request an interval, credible interval or coverage statement for the
    /// whole procedure.
    ///
    /// # Errors
    /// Always `cell_not_licensed`: no coverage record measures the whole
    /// procedure (assignment, outcome structure, monitoring and stopping
    /// together), and a guarantee for components does not license the pairing.
    pub fn interval(&self) -> Result<(), AdaptiveInferenceRefusal> {
        Err(AdaptiveInferenceRefusal {
            code: antecedent_core::reason_code!("cell_not_licensed"),
            detail: "adaptive_inference.interval_closed".to_owned(),
            message: format!(
                "no interval, credible interval or coverage statement is licensed for the procedure {}: whole-procedure coverage under its assignment, outcome structure, monitoring strategy and stopping rule has not been measured; the point values and uncalibrated diagnostics are the result",
                self.contract.cell().identifier()
            ),
        })
    }

    /// Export the monitoring contract for a paradigm.
    ///
    /// # Errors
    /// A paradigm the cell is not licensed for, or a contract that does not seal.
    pub fn export_contract(
        &self,
        paradigm: InferenceParadigm,
        artifact_id: &str,
    ) -> Result<Vec<u8>, AdaptiveInferenceRefusal> {
        AdaptiveMonitoringContractArtifact::to_bytes(&self.contract, paradigm, artifact_id)
            .map_err(|error| artifact_refusal(&error.to_string(), error.refusal()))
    }

    /// Export a result bound to its monitoring contract.
    ///
    /// # Errors
    /// A refused request or a result that does not seal.
    pub fn export_result(
        &self,
        request: &AdaptiveResultSpec,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, AdaptiveInferenceRefusal> {
        AdaptiveInferenceResultArtifact::to_bytes(&self.contract, request, artifact_id, ctx)
            .map_err(|error| artifact_refusal(&error.to_string(), error.refusal()))
    }
}

fn artifact_refusal(
    text: &str,
    refusal: Option<(&'static str, String, String)>,
) -> AdaptiveInferenceRefusal {
    match refusal {
        Some((code, detail, message)) => AdaptiveInferenceRefusal { code, detail, message },
        None => AdaptiveInferenceRefusal {
            code: antecedent_core::reason_code!("invalid_argument"),
            detail: "adaptive_inference.input_invalid".to_owned(),
            message: text.to_owned(),
        },
    }
}

/// A consumed monitoring contract.
#[derive(Clone, Debug)]
pub struct ConsumedMonitoringContract {
    artifact: AdaptiveMonitoringContractArtifact,
}

impl ConsumedMonitoringContract {
    /// The procedure the inference used.
    #[must_use]
    pub const fn procedure(&self) -> ProcedureCell {
        self.artifact.procedure()
    }

    /// The paradigm the inference belongs to.
    #[must_use]
    pub const fn paradigm(&self) -> InferenceParadigm {
        self.artifact.paradigm()
    }

    /// The stopping rule the inference used.
    #[must_use]
    pub fn stopping_rule(&self) -> Option<StoppingRuleDeclaration> {
        self.artifact.stopping_rule()
    }

    /// The replayed contract.
    #[must_use]
    pub const fn contract(&self) -> &AdaptiveDesignContract {
        self.artifact.contract()
    }

    /// Identity (hex) of the artifact.
    #[must_use]
    pub fn identity(&self) -> &str {
        self.artifact.identity()
    }
}

/// Consume a monitoring contract artifact by replay.
///
/// `retained_identity` is an identity (hex) the consumer kept independently of
/// the bytes.
///
/// # Errors
/// Corruption, a missing or inconsistent contract element, a paradigm the
/// procedure cell is not licensed for, a stored value that does not replay, or
/// an identity that does not match.
pub fn consume_adaptive_monitoring_contract_artifact(
    bytes: &[u8],
    retained_identity: Option<&str>,
) -> Result<ConsumedMonitoringContract, AdaptiveInferenceRefusal> {
    AdaptiveMonitoringContractArtifact::from_bytes(bytes, retained_identity)
        .map(|artifact| ConsumedMonitoringContract { artifact })
        .map_err(|error| artifact_refusal(&error.to_string(), error.refusal()))
}

/// A consumed adaptive inference result.
#[derive(Clone, Debug)]
pub struct ConsumedAdaptiveResult {
    artifact: AdaptiveInferenceResultArtifact,
}

impl ConsumedAdaptiveResult {
    /// The procedure the inference used.
    #[must_use]
    pub const fn procedure(&self) -> ProcedureCell {
        self.artifact.contract().procedure()
    }

    /// The paradigm the inference belongs to.
    #[must_use]
    pub const fn paradigm(&self) -> InferenceParadigm {
        self.artifact.contract().paradigm()
    }

    /// The stopping rule the inference used.
    #[must_use]
    pub fn stopping_rule(&self) -> Option<StoppingRuleDeclaration> {
        self.artifact.contract().stopping_rule()
    }

    /// The recomputed result.
    #[must_use]
    pub const fn result(&self) -> &AdaptiveResult {
        self.artifact.result()
    }

    /// Identity (hex) of the artifact.
    #[must_use]
    pub fn identity(&self) -> &str {
        self.artifact.identity()
    }
}

/// Consume an adaptive inference result artifact by replay and recomputation.
///
/// # Errors
/// Corruption, a contract refusal, a stored value that does not replay, or an
/// identity that does not match the retained one.
pub fn consume_adaptive_inference_result_artifact(
    bytes: &[u8],
    retained_identity: Option<&str>,
    ctx: &ExecutionContext,
) -> Result<ConsumedAdaptiveResult, AdaptiveInferenceRefusal> {
    AdaptiveInferenceResultArtifact::from_bytes(bytes, retained_identity, ctx)
        .map(|artifact| ConsumedAdaptiveResult { artifact })
        .map_err(|error| artifact_refusal(&error.to_string(), error.refusal()))
}
