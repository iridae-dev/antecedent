//! Inference from repeated categorical choices: prepare the choice-study contract, run one
//! inferential paradigm, derive the procedure's exact law, export and independently consume.
//!
//! The routes are the choice cells of the adaptive-procedure matrix. This module routes a request
//! through the layers that own each rule and adds no scientific rule of its own:
//!
//! 1. **Prepare** ([`ChoiceInferenceSession::prepare`]) validates the retained study
//!    (`antecedent_contract::adaptive_design::ChoiceDesignContract`): the frame and ordered
//!    options, every respondent's rows, each retained source's scope against the target and its
//!    rows against the respondents' (through the evidence ledger), the looks and the stopping
//!    rule. A missing or inconsistent element is a typed refusal and nothing is defaulted.
//! 2. **Analyse** ([`ChoiceInferenceSession::analyse`]) authorizes one paradigm for the cell and
//!    runs it: the pooled Dirichlet posterior, the hierarchical posterior of repeated choices,
//!    or the contract's stopping decision on either. Every result is point-only; interval and
//!    guarantee routes refuse with `not_supported` until coverage is measured.
//! 3. **Operating characteristics** ([`ChoiceInferenceSession::operating_characteristics`]) is
//!    the exact law of the contract's sequential procedure under a declared kernel.
//! 4. **Export / consume** write and replay `choice_inference_result_v1`, recomputing every
//!    value from the retained respondents and sources.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::choice_ranking::{
    ChoiceEvsiError, rank_hierarchical_studies, rank_studies_with_context,
};
pub use super::choice_ranking::{EvsiMethod, StudyCandidate, StudyRank};
use antecedent_contract::adaptive_design::{
    CellStatus, InferenceParadigm, ProcedureCell, procedure_status,
};
pub use antecedent_contract::adaptive_design::{
    ChoiceDesignContract, ChoiceDesignDeclaration, ChoiceLook, ChoiceRespondent,
    RetainedChoiceSource,
};
use antecedent_core::ExecutionContext;
pub use antecedent_estimate::choice_inference::{
    ChoiceKernel, ChoiceOperatingCharacteristics, ChoiceProcedureSpec, ChoiceStoppingRule,
};
use antecedent_estimate::choice_inference::{
    choice_operating_characteristics, contract_posterior, contract_procedure_spec,
    qualify_contract_prior, qualify_contract_prior_at_look,
};
pub use antecedent_io::choice_inference_result_artifact::ChoiceResultSpec as ChoiceRequest;
use antecedent_io::choice_inference_result_artifact::{
    ChoiceInferenceResultArtifact, ChoiceResult, ChoiceResultError, compute_choice_result,
    verify_retained_choice_sources,
};

/// One refusal as plain data: registered reason code, namespaced `choice_inference.*` detail,
/// and what would support the request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChoiceInferenceRefusal {
    /// Registered runtime reason code.
    pub code: &'static str,
    /// Namespaced detail.
    pub detail: String,
    /// What was refused and what would support it.
    pub message: String,
}

impl std::fmt::Display for ChoiceInferenceRefusal {
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

impl std::error::Error for ChoiceInferenceRefusal {}

impl From<ChoiceResultError> for ChoiceInferenceRefusal {
    fn from(error: ChoiceResultError) -> Self {
        Self { code: error.code, detail: error.detail, message: error.message }
    }
}

fn from_external(refusal: &antecedent_contract::ExternalRefusal) -> ChoiceInferenceRefusal {
    ChoiceInferenceRefusal {
        code: refusal.code,
        detail: refusal.detail.clone(),
        message: refusal.remedy.unwrap_or("").to_owned(),
    }
}

fn from_estimator(
    error: antecedent_estimate::adaptive_inference::AdaptiveInferenceError,
) -> ChoiceInferenceRefusal {
    ChoiceInferenceRefusal {
        code: error.code,
        detail: error.detail.to_owned(),
        message: error.message,
    }
}

fn artifact_refusal(
    text: &str,
    refusal: Option<(&'static str, String, String)>,
) -> ChoiceInferenceRefusal {
    match refusal {
        Some((code, detail, message)) => ChoiceInferenceRefusal { code, detail, message },
        None => ChoiceInferenceRefusal {
            code: antecedent_core::reason_code!("invalid_argument"),
            detail: "choice_inference.input_invalid".to_owned(),
            message: text.to_owned(),
        },
    }
}

/// A result bound to the contract and request it was computed from.
#[derive(Clone, Debug, PartialEq)]
pub struct ChoiceInferenceOutcome {
    /// The request.
    pub request: ChoiceRequest,
    /// The procedure cell the study ran under.
    pub procedure: ProcedureCell,
    /// The computed values and their method.
    pub result: ChoiceResult,
    /// Identity (hex) of the contract the result was computed under.
    pub contract_identity: String,
}

/// A prepared choice-study analysis with current source standing.
#[derive(Debug)]
pub struct ChoiceInferenceSession {
    contract: ChoiceDesignContract,
    _lease: std::sync::Arc<antecedent_core::execution::AdmissionLease>,
}

impl ChoiceInferenceSession {
    /// Validate the retained choice study.
    ///
    /// # Errors
    /// One refusal per missing element and per inconsistency.
    pub fn prepare(
        declaration: ChoiceDesignDeclaration,
    ) -> Result<Self, Vec<ChoiceInferenceRefusal>> {
        Self::prepare_with_context(declaration, &ExecutionContext::production_default(0))
    }

    /// Validate a choice study and replay retained native posteriors under the caller's context.
    ///
    /// # Errors
    /// Contract, source replay, cancellation or resource refusals.
    pub fn prepare_with_context(
        declaration: ChoiceDesignDeclaration,
        ctx: &ExecutionContext,
    ) -> Result<Self, Vec<ChoiceInferenceRefusal>> {
        let refusal = |code, detail: &str, message: String| {
            vec![ChoiceInferenceRefusal { code, detail: detail.into(), message }]
        };
        let rows = declaration
            .respondents
            .iter()
            .try_fold(0usize, |n, r| {
                r.counts.iter().try_fold(n, |sum, c| sum.checked_add(*c as usize))
            })
            .and_then(|n| {
                declaration
                    .retained_sources
                    .iter()
                    .try_fold(n, |sum, source| sum.checked_add(source.observation_ids.len()))
            });
        let proof_bytes = declaration.retained_sources.iter().try_fold(0usize, |n, source| {
            n.checked_add(source.posterior_proof.as_ref().map_or(0, |p| p.artifact.len()))
        });
        let string_bytes = declaration
            .respondents
            .iter()
            .try_fold(0usize, |n, r| {
                n.checked_add(r.id.len())
                    .and_then(|v| v.checked_add(r.counts.len().saturating_mul(4)))
            })
            .and_then(|n| {
                declaration.retained_sources.iter().try_fold(n, |sum, source| {
                    source.observation_ids.iter().try_fold(sum, |v, id| v.checked_add(id.len()))
                })
            });
        let quote = rows
            .zip(proof_bytes)
            .zip(string_bytes)
            .and_then(|((rows, proofs), strings)| {
                rows.checked_mul(160)
                    .and_then(|n| proofs.checked_mul(2).and_then(|p| n.checked_add(p)))
                    .and_then(|n| strings.checked_mul(2).and_then(|v| n.checked_add(v)))
            })
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| {
                refusal(
                    antecedent_core::reason_code!("execution_resource_limit"),
                    "choice_inference.admission_overflow",
                    "choice source admission size overflow".into(),
                )
            })?;
        let lease = ctx.acquire_admission(0, quote).map(std::sync::Arc::new).map_err(|e| {
            if ctx.cancellation.is_cancelled() {
                refusal(
                    antecedent_core::reason_code!("execution_cancelled"),
                    "choice_inference.cancelled",
                    e.to_string(),
                )
            } else {
                refusal(
                    antecedent_core::reason_code!("execution_resource_limit"),
                    "choice_inference.resource_limit",
                    e.to_string(),
                )
            }
        })?;
        let contract = ChoiceDesignContract::new(declaration)
            .map_err(|error| error.refusals.iter().map(from_external).collect::<Vec<_>>())?;
        verify_retained_choice_sources(&contract, ctx)
            .map_err(|e| vec![artifact_refusal(&e.to_string(), e.refusal())])?;
        Ok(Self { contract, _lease: lease })
    }

    /// Withdraw a retained source from this session's current evidence ledger.
    /// Previously exported artifacts remain historical; they cannot authorize new results here.
    ///
    /// # Errors
    /// Unknown source, empty reason or a refused ledger operation.
    pub fn withdraw_source(
        &mut self,
        label: &str,
        reason: &str,
    ) -> Result<(), ChoiceInferenceRefusal> {
        self.contract.withdraw_retained_source(label, reason).map_err(|r| from_external(&r))
    }

    /// Consume a retained result only if its contract matches this session and all sources remain active.
    ///
    /// # Errors
    /// A withdrawn source, another contract or a failed artifact replay.
    pub fn consume_current(
        &self,
        bytes: &[u8],
        retained_identity: &str,
        ctx: &ExecutionContext,
    ) -> Result<ConsumedChoiceResult, ChoiceInferenceRefusal> {
        self.contract.ensure_current().map_err(|r| from_external(&r))?;
        let result = consume_choice_inference_result_artifact(bytes, Some(retained_identity), ctx)?;
        if result.contract().identity() != self.contract.identity() {
            return Err(ChoiceInferenceRefusal {
                code: antecedent_core::reason_code!("invalid_argument"),
                detail: "choice_inference.current_contract_mismatch".into(),
                message: "the retained result belongs to another choice contract".into(),
            });
        }
        Ok(result)
    }

    /// The validated contract.
    #[must_use]
    pub const fn contract(&self) -> &ChoiceDesignContract {
        &self.contract
    }

    /// Identity (hex) of the contract.
    #[must_use]
    pub fn contract_identity(&self) -> String {
        self.contract.identity().to_hex()
    }

    /// What the procedure matrix does with this study's cell.
    #[must_use]
    pub fn cell_status(&self) -> CellStatus {
        procedure_status(&self.contract.cell())
    }

    /// Run the inferential paradigm a request names.
    ///
    /// # Errors
    /// A paradigm the cell is not supported for, a request of the wrong outcome structure, a
    /// numerical refusal, or cancellation.
    pub fn analyse(
        &self,
        request: &ChoiceRequest,
        ctx: &ExecutionContext,
    ) -> Result<ChoiceInferenceOutcome, ChoiceInferenceRefusal> {
        let result = compute_choice_result(&self.contract, request, ctx)?;
        Ok(ChoiceInferenceOutcome {
            request: request.clone(),
            procedure: self.contract.cell(),
            result,
            contract_identity: self.contract_identity(),
        })
    }

    /// The exact law of the contract's sequential procedure under `kernel`: probabilities of
    /// selecting each option, of stopping at each look and of reaching the horizon, and the
    /// expected enrolment.
    ///
    /// # Errors
    /// A repeated-choice contract, more than four options, a kernel or rule outside its domain,
    /// a size bound exceeded, or cancellation.
    pub fn operating_characteristics(
        &self,
        kernel: &ChoiceKernel,
        ctx: &ExecutionContext,
    ) -> Result<ChoiceOperatingCharacteristics, ChoiceInferenceRefusal> {
        self.contract.ensure_current().map_err(|r| from_external(&r))?;
        if qualify_contract_prior(&self.contract).map_err(from_estimator)?.has_conflict() {
            return Err(ChoiceInferenceRefusal {
                code: antecedent_core::reason_code!("choice_evidence_conflict"),
                detail: "choice_inference.source_conflict".into(),
                message: "resolve conflicting retained sources before evaluating the procedure"
                    .into(),
            });
        }
        let spec = contract_procedure_spec(&self.contract).map_err(from_estimator)?;
        choice_operating_characteristics(&spec, kernel, ctx).map_err(from_estimator)
    }

    /// Rank candidate additional studies by the expected value of sample information about the
    /// pooled posterior at `look`, net of cost. `values` is the value of each option when
    /// chosen, per unit share.
    ///
    /// # Errors
    /// A repeated-choice contract, a look the contract does not declare, values or candidates
    /// outside their domain, or cancellation.
    pub fn rank_studies(
        &self,
        look: usize,
        values: &[f64],
        candidates: &[StudyCandidate],
        seed: u64,
        ctx: &ExecutionContext,
    ) -> Result<Vec<StudyRank>, ChoiceInferenceRefusal> {
        self.contract
            .admit_paradigm(InferenceParadigm::BayesianPosterior)
            .map_err(|r| from_external(&r))?;
        if qualify_contract_prior_at_look(&self.contract, look)
            .map_err(from_estimator)?
            .has_conflict()
        {
            return Err(ChoiceInferenceRefusal {
                code: antecedent_core::reason_code!("choice_evidence_conflict"),
                detail: "choice_inference.source_conflict".into(),
                message:
                    "resolve conflict between retained and current evidence before ranking studies"
                        .into(),
            });
        }
        let posterior = contract_posterior(&self.contract, look, ctx).map_err(from_estimator)?;
        rank_studies_with_context(&posterior.alpha, values, candidates, seed, ctx).map_err(
            |e: ChoiceEvsiError| ChoiceInferenceRefusal {
                code: e.registered_code(),
                detail: "choice_inference.ranking_invalid".to_owned(),
                message: e.to_string(),
            },
        )
    }

    /// Rank one- or two-choice studies of one new independent respondent under a retained
    /// hierarchical posterior. The respondent's choices remain dependent.
    ///
    /// # Errors
    /// A different outcome or request, an unsupported cell, unsupported candidates, failed
    /// quadrature, cancellation or resource refusal.
    pub fn rank_hierarchical_studies(
        &self,
        request: &ChoiceRequest,
        values: &[f64],
        candidates: &[StudyCandidate],
        ctx: &ExecutionContext,
    ) -> Result<Vec<StudyRank>, ChoiceInferenceRefusal> {
        let ChoiceRequest::Hierarchical {
            look,
            shapes,
            log_concentration_mean,
            log_concentration_sd,
        } = request
        else {
            return Err(ChoiceInferenceRefusal {
                code: antecedent_core::reason_code!("invalid_argument"),
                detail: "choice_inference.route_outcome_mismatch".into(),
                message: "hierarchical ranking needs a hierarchical posterior request".into(),
            });
        };
        if shapes != &self.contract.declaration().initial_prior {
            return Err(ChoiceInferenceRefusal { code: antecedent_core::reason_code!("invalid_argument"), detail: "choice_inference.prior_mismatch".into(), message: "the hierarchical population-share prior must match the study's declared initial prior".into() });
        }
        if self.contract.cell().outcome
            != antecedent_contract::adaptive_design::OutcomeStructure::RepeatedChoice
        {
            return Err(ChoiceInferenceRefusal {
                code: antecedent_core::reason_code!("invalid_argument"),
                detail: "choice_inference.route_outcome_mismatch".into(),
                message: "hierarchical ranking requires repeated-choice observations".into(),
            });
        }
        self.contract
            .admit_paradigm(InferenceParadigm::BayesianPosterior)
            .map_err(|r| from_external(&r))?;
        let rows = self.contract.respondents_at(*look as usize).map_err(|r| from_external(&r))?;
        let quote = rows
            .len()
            .checked_mul(
                std::mem::size_of::<Vec<u32>>().saturating_add(shapes.len().saturating_mul(4)),
            )
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| ChoiceInferenceRefusal {
                code: antecedent_core::reason_code!("execution_resource_limit"),
                detail: "choice_inference.ranking_admission".into(),
                message: "hierarchical ranking admission overflow".into(),
            })?;
        let _lease = ctx.acquire_admission(0, quote).map_err(|e| ChoiceInferenceRefusal {
            code: if ctx.cancellation.is_cancelled() {
                antecedent_core::reason_code!("execution_cancelled")
            } else {
                antecedent_core::reason_code!("execution_resource_limit")
            },
            detail: "choice_inference.ranking_admission".into(),
            message: e.to_string(),
        })?;
        let counts: Vec<Vec<u32>> = rows.iter().map(|r| r.counts.clone()).collect();
        let prior = antecedent_estimate::choice_inference::HierarchicalChoicePrior {
            shapes: shapes.clone(),
            log_concentration_mean: *log_concentration_mean,
            log_concentration_sd: *log_concentration_sd,
        };
        rank_hierarchical_studies(&prior, &counts, values, candidates, ctx).map_err(|e| {
            ChoiceInferenceRefusal {
                code: e.registered_code(),
                detail: "choice_inference.ranking_invalid".into(),
                message: e.to_string(),
            }
        })
    }

    /// Request an interval or coverage statement without naming a result.
    ///
    /// # Errors
    /// Always `not_supported`: a coverage statement belongs to one measured cell and one
    /// named result. Ask for a result's interval with [`Self::measured_interval`].
    pub fn interval(&self) -> Result<(), ChoiceInferenceRefusal> {
        Err(ChoiceInferenceRefusal {
            code: antecedent_core::reason_code!("not_supported"),
            detail: "choice_inference.interval_closed".to_owned(),
            message: format!(
                "no interval or coverage statement is supported for the choice procedure {} as an unqualified request; request the interval of a result with measured_interval, which is supported only for measured cells",
                self.contract.cell().identifier()
            ),
        })
    }

    /// The interval of one result of a measured choice cell, bound to currently attesting
    /// coverage. The result is sealed and replayed first.
    ///
    /// # Errors
    /// A result the contract cannot produce, a study outside the measured cells
    /// (`not_supported`), or coverage evidence that is absent or no longer attesting.
    pub fn measured_interval(
        &self,
        request: &ChoiceRequest,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<antecedent_io::measured_inference::MeasuredInference, ChoiceInferenceRefusal> {
        let bytes = self.export_result(request, artifact_id, ctx)?;
        ChoiceInferenceResultArtifact::measure_interval(&bytes, ctx).map_err(|error| match error {
            antecedent_io::IoError::Refused { code, message } => {
                let (detail, text) = message
                    .split_once(": ")
                    .unwrap_or(("choice_inference.interval_closed", message.as_str()));
                ChoiceInferenceRefusal { code, detail: detail.to_owned(), message: text.to_owned() }
            }
            other => ChoiceInferenceRefusal {
                code: antecedent_core::reason_code!("invalid_argument"),
                detail: "choice_inference.input_invalid".to_owned(),
                message: other.to_string(),
            },
        })
    }

    /// Export a result bound to its contract.
    ///
    /// # Errors
    /// A refused request or a result that does not seal.
    pub fn export_result(
        &self,
        request: &ChoiceRequest,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, ChoiceInferenceRefusal> {
        ChoiceInferenceResultArtifact::to_bytes(&self.contract, request, artifact_id, ctx)
            .map_err(|error| artifact_refusal(&error.to_string(), error.refusal()))
    }
}

/// Reconstruct a native retained posterior source, retaining its independently held identity.
///
/// # Errors
/// A corrupt, unbound, incompatible or unsupported result.
pub fn retained_source_from_artifact(
    bytes: &[u8],
    expected_identity: &str,
    label: &str,
    discount: f64,
    ctx: &ExecutionContext,
) -> Result<RetainedChoiceSource, ChoiceInferenceRefusal> {
    antecedent_io::choice_inference_result_artifact::retained_choice_source_from_artifact(
        bytes,
        expected_identity,
        label,
        discount,
        ctx,
    )
    .map_err(|e| artifact_refusal(&e.to_string(), e.refusal()))
}

/// A consumed choice inference result.
#[derive(Clone, Debug)]
pub struct ConsumedChoiceResult {
    artifact: ChoiceInferenceResultArtifact,
}

impl ConsumedChoiceResult {
    /// The procedure cell the study ran under.
    #[must_use]
    pub fn procedure(&self) -> ProcedureCell {
        self.artifact.contract().cell()
    }

    /// The replayed contract.
    #[must_use]
    pub const fn contract(&self) -> &ChoiceDesignContract {
        self.artifact.contract()
    }

    /// The request.
    #[must_use]
    pub const fn request(&self) -> &ChoiceRequest {
        self.artifact.spec()
    }

    /// The recomputed result.
    #[must_use]
    pub const fn result(&self) -> &ChoiceResult {
        self.artifact.result()
    }

    /// Identity (hex) of the artifact.
    #[must_use]
    pub fn identity(&self) -> &str {
        self.artifact.identity()
    }
}

/// Consume a choice inference result artifact by replay and recomputation.
///
/// # Errors
/// Corruption, a contract refusal, a stored value that does not replay, or an identity that does
/// not match the retained one.
pub fn consume_choice_inference_result_artifact(
    bytes: &[u8],
    retained_identity: Option<&str>,
    ctx: &ExecutionContext,
) -> Result<ConsumedChoiceResult, ChoiceInferenceRefusal> {
    ChoiceInferenceResultArtifact::from_bytes(bytes, retained_identity, ctx)
        .map(|artifact| ConsumedChoiceResult { artifact })
        .map_err(|error| artifact_refusal(&error.to_string(), error.refusal()))
}
