//! Choice inference result artifact (`choice_inference_result_v1`).
//!
//! One inferential result from a choice study, sealed with the analysis contract it was computed
//! under:
//!
//! * the embedded choice-study declaration (frame, ordered options, respondents' rows, retained
//!   sources with their discounts and transfer reasons, looks, stopping rule, procedure cell);
//! * the request: which look, which route and its prior parameters; and
//! * the stored values, the method, the point-only claim, the **unmeasured** calibration status
//!   and the closed guarantee.
//!
//! A consumer trusts none of the stored values. It rebuilds the contract from the declaration
//! (so a missing element, a source on rows the respondents also contribute, a source of another
//! frame without a declared transfer, or a paradigm the cell is not supported for is refused),
//! requires the request's paradigm and outcome structure to be the cell's, and recomputes the
//! result with the estimators. A resealed edit of a count, a discount, the stopping rule or a
//! stored value is refused, and a consumer that retained the identity independently refuses a
//! consistently resealed result of another study.
//!
//! What this does **not** say: no interval, credible interval or coverage statement (closed:
//! `not_supported`), no frequentist error rate for the stopping rule, and no claim that
//! stated choices identify behaviour or a causal effect.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::adaptive_design::{
    AssignmentFamily, ChoiceDesignContract, ChoiceDesignDeclaration, ChoiceLook, ChoiceRespondent,
    InferenceParadigm, MonitoringStrategy, OutcomeStructure, ProcedureCell, RetainedChoiceProof,
    RetainedChoiceSource, StoppingRuleKind,
};
use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_estimate::adaptive_inference::AdaptiveInferenceError;
use antecedent_estimate::choice_inference::{
    HierarchicalChoicePrior, choice_posterior, choice_stopping_decision, contract_pooled_counts,
    contract_stopping_rule, hierarchical_choice_posterior, hierarchical_stopping_decision,
    qualify_contract_prior_at_look,
};
use antecedent_io_core::adaptive_monitoring_contract_artifact::{
    AdaptiveMonitoringContractError, RuleWire, rule_from_wire, rule_to_wire,
};
use antecedent_io_core::replay_tolerance::{ReplayNonFinite, ReplayScale, ReplayTolerance};
use serde::{Deserialize, Serialize};

/// The artifact major version this reader writes and accepts.
pub const CHOICE_INFERENCE_RESULT_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const CHOICE_INFERENCE_RESULT_ARTIFACT_FEATURE: &str = "choice_inference_result_v1";
/// Calibration status: no coverage record measures any choice route.
pub const CHOICE_INFERENCE_CALIBRATION: &str = "unmeasured";
/// Most bytes an artifact may occupy.
pub const MAX_CHOICE_INFERENCE_RESULT_BYTES: usize = 256 * 1024 * 1024;
/// Relative tolerance when comparing a stored value with its recomputation.
pub const CHOICE_RESULT_REPLAY_TOLERANCE: f64 = 1e-9;

const ARTIFACT_KIND: &str = crate::artifact_kinds::CHOICE_INFERENCE_RESULT;
const META_SECTION: &str = "choice_inference_result_meta";

crate::sectioned_artifact_error!(
    /// Why a choice inference result artifact was refused or could not be produced.
    ChoiceInferenceResultArtifactError,
    "choice inference result",
    "choice_inference",
    "result",
    true
);

type Error = ChoiceInferenceResultArtifactError;

fn malformed(text: impl Into<String>) -> Error {
    Error::Malformed(text.into())
}

// -- the request ------------------------------------------------------------------------------

/// What is asked of a choice study's retained history.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ChoiceResultSpec {
    /// The pooled Dirichlet posterior of a categorical-choice outcome after `look`, on the prior
    /// the contract's retained sources qualify.
    Posterior {
        /// Look index.
        look: u32,
    },
    /// The contract's posterior stopping rule on that posterior after `look`.
    Stopping {
        /// Look index.
        look: u32,
    },
    /// The hierarchical posterior of a repeated-choice outcome after `look`.
    Hierarchical {
        /// Look index.
        look: u32,
        /// Dirichlet shapes of the population shares.
        shapes: Vec<f64>,
        /// Mean of `ln kappa`.
        log_concentration_mean: f64,
        /// Standard deviation of `ln kappa`.
        log_concentration_sd: f64,
    },
    /// The contract's expected-loss rule on the hierarchical posterior after `look`.
    HierarchicalStopping {
        /// Look index.
        look: u32,
        /// Dirichlet shapes of the population shares.
        shapes: Vec<f64>,
        /// Mean of `ln kappa`.
        log_concentration_mean: f64,
        /// Standard deviation of `ln kappa`.
        log_concentration_sd: f64,
    },
}

impl ChoiceResultSpec {
    /// The paradigm this request belongs to.
    #[must_use]
    pub const fn paradigm(&self) -> InferenceParadigm {
        match self {
            Self::Posterior { .. } | Self::Hierarchical { .. } => {
                InferenceParadigm::BayesianPosterior
            }
            Self::Stopping { .. } | Self::HierarchicalStopping { .. } => {
                InferenceParadigm::DecisionTheoreticStopping
            }
        }
    }

    /// The outcome structure the request is defined for.
    #[must_use]
    pub const fn outcome(&self) -> OutcomeStructure {
        match self {
            Self::Posterior { .. } | Self::Stopping { .. } => OutcomeStructure::CategoricalChoice,
            Self::Hierarchical { .. } | Self::HierarchicalStopping { .. } => {
                OutcomeStructure::RepeatedChoice
            }
        }
    }

    /// The look the request is about.
    #[must_use]
    pub const fn look(&self) -> u32 {
        match self {
            Self::Posterior { look }
            | Self::Stopping { look }
            | Self::Hierarchical { look, .. }
            | Self::HierarchicalStopping { look, .. } => *look,
        }
    }
}

/// The values of one result, in the route's fixed order.
#[derive(Clone, Debug, PartialEq)]
pub struct ChoiceResult {
    /// The paradigm.
    pub paradigm: InferenceParadigm,
    /// The method that produced the values.
    pub method: &'static str,
    /// Values in the order documented on [`compute_choice_result`].
    pub values: Vec<f64>,
    /// Detail of the closed guarantee that goes with the values.
    pub closed_guarantee: &'static str,
}

/// A refusal to compute a result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChoiceResultError {
    /// Registered reason code.
    pub code: &'static str,
    /// Namespaced detail.
    pub detail: String,
    /// What was refused and what would support it.
    pub message: String,
}

impl std::fmt::Display for ChoiceResultError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.detail, self.message)
    }
}

impl std::error::Error for ChoiceResultError {}

impl From<AdaptiveInferenceError> for ChoiceResultError {
    fn from(error: AdaptiveInferenceError) -> Self {
        Self { code: error.code, detail: error.detail.to_owned(), message: error.message }
    }
}

fn contract_refusal(refusal: &antecedent_contract::ExternalRefusal) -> ChoiceResultError {
    ChoiceResultError {
        code: refusal.code,
        detail: refusal.detail.clone(),
        message: refusal.remedy.unwrap_or("").to_owned(),
    }
}

fn request_invalid(detail: &str, message: &str) -> ChoiceResultError {
    ChoiceResultError {
        code: antecedent_core::reason_code!("invalid_argument"),
        detail: detail.to_owned(),
        message: message.to_owned(),
    }
}

// -- computation ------------------------------------------------------------------------------

fn hierarchical_inputs(
    contract: &ChoiceDesignContract,
    look: usize,
    shapes: &[f64],
    mean: f64,
    sd: f64,
) -> Result<(HierarchicalChoicePrior, Vec<Vec<u32>>), ChoiceResultError> {
    if shapes != contract.declaration().initial_prior.as_slice() {
        return Err(request_invalid(
            "choice_inference.prior_mismatch",
            "the hierarchical population-share prior must match the study's declared initial prior",
        ));
    }
    let respondents = contract
        .respondents_at(look)
        .map_err(|r| contract_refusal(&r))?
        .iter()
        .map(|r| r.counts.clone())
        .collect();
    Ok((
        HierarchicalChoicePrior {
            shapes: shapes.to_vec(),
            log_concentration_mean: mean,
            log_concentration_sd: sd,
        },
        respondents,
    ))
}

/// Compute the result a request asks of a contract.
///
/// The route is authorized first: the paradigm must be one the contract's cell is supported for,
/// and the request's outcome structure must be the cell's. Value order:
///
/// * Posterior: per option `alpha, mean, probability_best, expected_loss`, then the observations
///   borrowed from retained sources, the number of sources flagged as conflicting and the
///   largest conflict log Bayes factor.
/// * Stopping: `best_option, probability_best, expected_loss, stop`.
/// * Hierarchical: per option `mean, probability_best, expected_loss`, then
///   `log_concentration_mean, log_concentration_sd, correlation_mean, log_marginal_likelihood`.
/// * Hierarchical stopping: `best_option, probability_best, expected_loss, stop`.
///
/// # Errors
/// A refused authorization, a request outside the contract, a numerical refusal, or
/// cancellation.
#[allow(clippy::too_many_lines, reason = "the result preserves the complete choice computation")]
pub fn compute_choice_result(
    contract: &ChoiceDesignContract,
    spec: &ChoiceResultSpec,
    ctx: &ExecutionContext,
) -> Result<ChoiceResult, ChoiceResultError> {
    let paradigm = spec.paradigm();
    contract.admit_paradigm(paradigm).map_err(|r| contract_refusal(&r))?;
    verify_retained_choice_sources(contract, ctx).map_err(|e| {
        let (code, detail, message) = e.refusal().unwrap_or((
            antecedent_core::reason_code!("invalid_argument"),
            "choice_inference.source_invalid".into(),
            e.to_string(),
        ));
        ChoiceResultError { code, detail, message }
    })?;
    if contract.cell().outcome != spec.outcome() {
        return Err(request_invalid(
            "choice_inference.route_outcome_mismatch",
            "a pooled posterior is for one choice per respondent and a hierarchical posterior for repeated choices; request the route of the contract's outcome",
        ));
    }
    let look = spec.look() as usize;
    match spec {
        ChoiceResultSpec::Posterior { .. } | ChoiceResultSpec::Stopping { .. } => {
            let prior = qualify_contract_prior_at_look(contract, look)?;
            if matches!(spec, ChoiceResultSpec::Stopping { .. }) && prior.has_conflict() {
                return Err(ChoiceResultError { code: antecedent_core::reason_code!("choice_evidence_conflict"), detail: "choice_inference.source_conflict".into(), message: "the current look conflicts with retained evidence; resolve the conflict before a stopping decision".into() });
            }
            let counts = contract_pooled_counts(contract, look)?;
            let posterior = choice_posterior(&prior.alpha, &counts, ctx)?;
            if matches!(spec, ChoiceResultSpec::Stopping { .. }) {
                let decision =
                    choice_stopping_decision(&posterior, contract_stopping_rule(contract)?)?;
                return Ok(ChoiceResult {
                    paradigm,
                    method: "dirichlet_posterior_stopping",
                    values: vec![
                        decision.best_option as f64,
                        decision.probability_best,
                        decision.expected_loss,
                        f64::from(u8::from(decision.stop)),
                    ],
                    closed_guarantee: decision.guarantee.detail,
                });
            }
            let mut values: Vec<f64> = (0..posterior.alpha.len())
                .flat_map(|k| {
                    [
                        posterior.alpha[k],
                        posterior.mean[k],
                        posterior.probability_best[k],
                        posterior.expected_loss[k],
                    ]
                })
                .collect();
            values.push(prior.borrowed_observations);
            values.push(prior.sources.iter().filter(|s| s.conflicting).count() as f64);
            values.push(
                prior
                    .sources
                    .iter()
                    .map(|s| s.conflict_log_bayes_factor)
                    .fold(f64::NEG_INFINITY, f64::max)
                    .max(0.0),
            );
            Ok(ChoiceResult {
                paradigm,
                method: "dirichlet_multinomial_conjugate_posterior",
                values,
                closed_guarantee: posterior.guarantee.detail,
            })
        }
        ChoiceResultSpec::Hierarchical {
            shapes,
            log_concentration_mean,
            log_concentration_sd,
            ..
        }
        | ChoiceResultSpec::HierarchicalStopping {
            shapes,
            log_concentration_mean,
            log_concentration_sd,
            ..
        } => {
            let (prior, respondents) = hierarchical_inputs(
                contract,
                look,
                shapes,
                *log_concentration_mean,
                *log_concentration_sd,
            )?;
            let result = hierarchical_choice_posterior(&prior, &respondents, ctx)?;
            if matches!(spec, ChoiceResultSpec::HierarchicalStopping { .. }) {
                let decision =
                    hierarchical_stopping_decision(&result, contract_stopping_rule(contract)?)?;
                return Ok(ChoiceResult {
                    paradigm,
                    method: "hierarchical_dirichlet_multinomial_stopping",
                    values: vec![
                        decision.best_option as f64,
                        decision.probability_best,
                        decision.expected_loss,
                        f64::from(u8::from(decision.stop)),
                    ],
                    closed_guarantee: decision.guarantee.detail,
                });
            }
            let s = &result.summary;
            let mut values: Vec<f64> = (0..s.mean.len())
                .flat_map(|k| [s.mean[k], s.probability_best[k], s.expected_loss[k]])
                .collect();
            values.extend([
                s.log_concentration_mean,
                s.log_concentration_sd,
                s.correlation_mean,
                s.log_marginal_likelihood,
            ]);
            Ok(ChoiceResult {
                paradigm,
                method: "hierarchical_dirichlet_multinomial_posterior",
                values,
                closed_guarantee: result.guarantee.detail,
            })
        }
    }
}

// -- wire ------------------------------------------------------------------------------------

/// The request on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChoiceSpecWire {
    /// Route kind.
    pub kind: String,
    /// Look index.
    pub look: u32,
    /// Prior shapes as `f64` bits (hierarchical routes).
    pub shape_bits: Vec<u64>,
    /// `ln kappa` mean and sd as `f64` bits (hierarchical routes).
    pub parameter_bits: Vec<u64>,
}

/// A respondent on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RespondentWire {
    /// Identity.
    pub id: String,
    /// Choices of each option.
    pub counts: Vec<u32>,
}

/// A retained source on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceWire {
    /// Label.
    pub label: String,
    /// Ordered options.
    pub option_labels: Vec<String>,
    /// What a choice meant.
    pub meaning: String,
    /// Population.
    pub population: String,
    /// Counts as `f64` bits.
    pub count_bits: Vec<u64>,
    /// Row identities.
    pub observation_ids: Vec<String>,
    /// Discount as `f64` bits.
    pub discount_bits: u64,
    /// Declared transfer reason.
    pub transfer_rationale: Option<String>,
    /// Optional native posterior artifact and independently retained identity.
    #[serde(default)]
    pub posterior_artifact: Option<Vec<u8>>,
    /// Identity of the retained native posterior.
    #[serde(default)]
    pub posterior_identity: Option<String>,
}

/// The choice declaration on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChoiceDeclarationWire {
    /// Label.
    pub label: String,
    /// Assignment family name.
    pub assignment: String,
    /// Outcome structure name.
    pub outcome: String,
    /// Monitoring strategy name.
    pub monitoring: String,
    /// Stopping rule kind name.
    pub stopping: String,
    /// Ordered options.
    pub option_labels: Vec<String>,
    /// What a choice means.
    pub meaning: String,
    /// The respondents' population.
    pub population: String,
    /// Respondents in enrolment order.
    pub respondents: Vec<RespondentWire>,
    /// Initial prior shapes as `f64` bits.
    pub initial_prior_bits: Vec<u64>,
    /// Retained sources.
    pub sources: Vec<SourceWire>,
    /// Respondents enrolled at each look; absent when not retained.
    pub looks: Option<Vec<u32>>,
    /// Stopping rule; absent when not retained.
    pub stopping_rule: Option<RuleWire>,
}

/// Stored metadata of a result artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChoiceResultMeta {
    /// Artifact major version.
    pub version: u16,
    /// Feature marker.
    pub feature: String,
    /// The embedded declaration.
    pub declaration: ChoiceDeclarationWire,
    /// Contract identity (hex).
    pub contract_identity: String,
    /// The request.
    pub spec: ChoiceSpecWire,
    /// Method name.
    pub method: String,
    /// Stored values as `f64` bits.
    pub value_bits: Vec<u64>,
    /// Claim class: always `point_only`.
    pub claim: String,
    /// Calibration status: always `unmeasured`.
    pub calibration: String,
    /// Detail of the closed guarantee.
    pub closed_guarantee: String,
    /// BLAKE3 identity (hex) of everything above.
    pub identity: String,
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn reals(values: &[u64]) -> Vec<f64> {
    values.iter().map(|v| f64::from_bits(*v)).collect()
}

/// A declaration as its wire form.
#[must_use]
pub fn declaration_to_wire(d: &ChoiceDesignDeclaration) -> ChoiceDeclarationWire {
    ChoiceDeclarationWire {
        label: d.label.clone(),
        assignment: d.cell.assignment.as_str().to_owned(),
        outcome: d.cell.outcome.as_str().to_owned(),
        monitoring: d.cell.monitoring.as_str().to_owned(),
        stopping: d.cell.stopping.as_str().to_owned(),
        option_labels: d.option_labels.clone(),
        meaning: d.meaning.clone(),
        population: d.population.clone(),
        respondents: d
            .respondents
            .iter()
            .map(|r| RespondentWire { id: r.id.clone(), counts: r.counts.clone() })
            .collect(),
        initial_prior_bits: bits(&d.initial_prior),
        sources: d
            .retained_sources
            .iter()
            .map(|s| SourceWire {
                label: s.label.clone(),
                option_labels: s.option_labels.clone(),
                meaning: s.meaning.clone(),
                population: s.population.clone(),
                count_bits: bits(&s.counts),
                observation_ids: s.observation_ids.clone(),
                discount_bits: s.discount.to_bits(),
                transfer_rationale: s.transfer_rationale.clone(),
                posterior_artifact: s.posterior_proof.as_ref().map(|p| p.artifact.clone()),
                posterior_identity: s.posterior_proof.as_ref().map(|p| p.expected_identity.clone()),
            })
            .collect(),
        looks: d.looks.as_ref().map(|l| l.iter().map(|l| l.respondents).collect()),
        stopping_rule: d.stopping_rule.as_ref().map(rule_to_wire),
    }
}

/// A declaration from its wire form.
///
/// # Errors
/// A name that is not a procedure-matrix element, or a malformed rule.
pub fn declaration_from_wire(w: &ChoiceDeclarationWire) -> Result<ChoiceDesignDeclaration, Error> {
    let cell = ProcedureCell {
        assignment: AssignmentFamily::from_name(&w.assignment)
            .ok_or(Error::UnsupportedSemantics("assignment family"))?,
        outcome: OutcomeStructure::from_name(&w.outcome)
            .ok_or(Error::UnsupportedSemantics("outcome structure"))?,
        monitoring: MonitoringStrategy::from_name(&w.monitoring)
            .ok_or(Error::UnsupportedSemantics("monitoring strategy"))?,
        stopping: StoppingRuleKind::from_name(&w.stopping)
            .ok_or(Error::UnsupportedSemantics("stopping rule kind"))?,
    };
    if w.sources.iter().any(|s| s.posterior_artifact.is_some() != s.posterior_identity.is_some()) {
        return Err(Error::Malformed(
            "retained posterior artifact and identity must be declared together".into(),
        ));
    }
    let stopping_rule =
        w.stopping_rule.as_ref().map(rule_from_wire).transpose().map_err(|e| match e {
            AdaptiveMonitoringContractError::UnsupportedSemantics(what) => {
                Error::UnsupportedSemantics(what)
            }
            other => Error::Malformed(other.to_string()),
        })?;
    Ok(ChoiceDesignDeclaration {
        label: w.label.clone(),
        cell,
        option_labels: w.option_labels.clone(),
        meaning: w.meaning.clone(),
        population: w.population.clone(),
        respondents: w
            .respondents
            .iter()
            .map(|r| ChoiceRespondent { id: r.id.clone(), counts: r.counts.clone() })
            .collect(),
        initial_prior: reals(&w.initial_prior_bits),
        retained_sources: w
            .sources
            .iter()
            .map(|s| RetainedChoiceSource {
                label: s.label.clone(),
                option_labels: s.option_labels.clone(),
                meaning: s.meaning.clone(),
                population: s.population.clone(),
                counts: reals(&s.count_bits),
                observation_ids: s.observation_ids.clone(),
                discount: f64::from_bits(s.discount_bits),
                transfer_rationale: s.transfer_rationale.clone(),
                posterior_proof: match (&s.posterior_artifact, &s.posterior_identity) {
                    (Some(artifact), Some(expected_identity)) => Some(RetainedChoiceProof {
                        artifact: artifact.clone(),
                        expected_identity: expected_identity.clone(),
                    }),
                    _ => None,
                },
            })
            .collect(),
        looks: w.looks.as_ref().map(|l| l.iter().map(|r| ChoiceLook { respondents: *r }).collect()),
        stopping_rule,
    })
}

fn spec_to_wire(spec: &ChoiceResultSpec) -> ChoiceSpecWire {
    let (kind, shapes, parameters): (&str, &[f64], Vec<f64>) = match spec {
        ChoiceResultSpec::Posterior { .. } => ("posterior", &[], Vec::new()),
        ChoiceResultSpec::Stopping { .. } => ("stopping", &[], Vec::new()),
        ChoiceResultSpec::Hierarchical {
            shapes,
            log_concentration_mean,
            log_concentration_sd,
            ..
        } => ("hierarchical", shapes, vec![*log_concentration_mean, *log_concentration_sd]),
        ChoiceResultSpec::HierarchicalStopping {
            shapes,
            log_concentration_mean,
            log_concentration_sd,
            ..
        } => {
            ("hierarchical_stopping", shapes, vec![*log_concentration_mean, *log_concentration_sd])
        }
    };
    ChoiceSpecWire {
        kind: kind.to_owned(),
        look: spec.look(),
        shape_bits: bits(shapes),
        parameter_bits: bits(&parameters),
    }
}

/// A request from its wire form.
///
/// # Errors
/// A request kind that is not a route, or a hierarchical request without its two parameters.
pub fn spec_from_wire(w: &ChoiceSpecWire) -> Result<ChoiceResultSpec, Error> {
    let look = w.look;
    let hierarchical = || -> Result<(Vec<f64>, f64, f64), Error> {
        match w.parameter_bits[..] {
            [m, s] => Ok((reals(&w.shape_bits), f64::from_bits(m), f64::from_bits(s))),
            _ => Err(malformed("a hierarchical request carries two prior parameters")),
        }
    };
    Ok(match w.kind.as_str() {
        "posterior" => ChoiceResultSpec::Posterior { look },
        "stopping" => ChoiceResultSpec::Stopping { look },
        "hierarchical" => {
            let (shapes, m, s) = hierarchical()?;
            ChoiceResultSpec::Hierarchical {
                look,
                shapes,
                log_concentration_mean: m,
                log_concentration_sd: s,
            }
        }
        "hierarchical_stopping" => {
            let (shapes, m, s) = hierarchical()?;
            ChoiceResultSpec::HierarchicalStopping {
                look,
                shapes,
                log_concentration_mean: m,
                log_concentration_sd: s,
            }
        }
        _ => return Err(Error::UnsupportedSemantics("request kind")),
    })
}

fn compute_identity(meta: &ChoiceResultMeta) -> String {
    let mut writer = CanonicalWriter::new("antecedent.choice_inference.result");
    writer.word(u64::from(meta.version));
    writer.text(&meta.feature).text(&meta.contract_identity);
    let spec = &meta.spec;
    writer.text(&spec.kind).word(u64::from(spec.look));
    writer.words(&spec.shape_bits).words(&spec.parameter_bits);
    writer.text(&meta.method).words(&meta.value_bits);
    writer.text(&meta.claim).text(&meta.calibration).text(&meta.closed_guarantee);
    writer.finish_hex()
}

/// A stored result reproduces the recomputed one relative to the recomputed value (absolute
/// below one); an undefined (NaN) value must be undefined in both.
const REPLAY: ReplayTolerance = ReplayTolerance::new(
    CHOICE_RESULT_REPLAY_TOLERANCE,
    ReplayScale::ReplayedOrOne,
    ReplayNonFinite::NanMatchesNan,
);

fn result_refusal(error: &ChoiceResultError) -> Error {
    Error::refused(error.code, error.detail.clone(), error.message.clone())
}

fn contract_from(declaration: ChoiceDesignDeclaration) -> Result<ChoiceDesignContract, Error> {
    ChoiceDesignContract::new(declaration).map_err(|error| {
        let first = &error.refusals[0];
        Error::refused(first.code, first.detail.clone(), first.remedy.unwrap_or(""))
    })
}

fn check(
    meta: &ChoiceResultMeta,
    expected_identity: Option<&str>,
    ctx: &ExecutionContext,
) -> Result<ChoiceInferenceResultArtifact, Error> {
    if meta.version != CHOICE_INFERENCE_RESULT_ARTIFACT_VERSION {
        return Err(Error::UnsupportedVersion { version: meta.version });
    }
    crate::artifact_codec::require_semantics::<Error>(
        meta.feature == CHOICE_INFERENCE_RESULT_ARTIFACT_FEATURE,
        "feature marker",
    )?;
    if meta.claim != "point_only" || meta.calibration != CHOICE_INFERENCE_CALIBRATION {
        return Err(Error::UnsupportedSemantics("claim or calibration status"));
    }
    let contract = contract_from(declaration_from_wire(&meta.declaration)?)?;
    if contract.identity().to_hex() != meta.contract_identity {
        return Err(Error::IdentityMismatch { field: "contract_identity" });
    }
    let spec = spec_from_wire(&meta.spec)?;
    let recomputed =
        compute_choice_result(&contract, &spec, ctx).map_err(|error| result_refusal(&error))?;
    if recomputed.method != meta.method || recomputed.closed_guarantee != meta.closed_guarantee {
        return Err(Error::ResultMismatch("method or closed guarantee"));
    }
    let stored: Vec<f64> = meta.value_bits.iter().map(|b| f64::from_bits(*b)).collect();
    if stored.len() != recomputed.values.len()
        || stored.iter().zip(&recomputed.values).any(|(s, r)| !REPLAY.close(*s, *r))
    {
        return Err(Error::ResultMismatch("values"));
    }
    let identity = compute_identity(meta);
    if meta.identity != identity {
        return Err(Error::IdentityMismatch { field: "identity" });
    }
    if expected_identity.is_some_and(|expected| expected != identity) {
        return Err(Error::IdentityMismatch { field: "retained_identity" });
    }
    Ok(ChoiceInferenceResultArtifact { contract, spec, result: recomputed, identity })
}

/// Reconstruct a same-population pooled source from an independently identified native result.
/// The source's replayed ancestors and their discounts are retained, and its original observations
/// enter once. The base prior is not counted twice; ancestry is bounded by depth and artifact size.
///
/// # Errors
/// Corrupt, unbound, incompatible, excessively nested, conflicting, or non-posterior artifacts.
pub fn retained_choice_source_from_artifact(
    bytes: &[u8],
    expected_identity: &str,
    label: &str,
    discount: f64,
    ctx: &ExecutionContext,
) -> Result<RetainedChoiceSource, Error> {
    if bytes.len() > 16 * 1024 * 1024
        || expected_identity.trim().is_empty()
        || label.trim().is_empty()
        || !discount.is_finite()
        || !(0.0..=1.0).contains(&discount)
    {
        return Err(malformed("invalid retained choice source identity, discount or size"));
    }
    let quote = u64::try_from(bytes.len())
        .ok()
        .and_then(|n| n.checked_mul(8))
        .ok_or_else(|| malformed("retained posterior admission size overflow"))?;
    let _lease = ctx.acquire_admission(0, quote).map_err(|error| {
        if ctx.cancellation.is_cancelled() {
            Error::refused(
                antecedent_core::reason_code!("execution_cancelled"),
                "choice_inference.source_cancelled",
                error.to_string(),
            )
        } else {
            Error::refused(
                antecedent_core::reason_code!("execution_resource_limit"),
                "choice_inference.source_resource_limit",
                error.to_string(),
            )
        }
    })?;
    let meta = decode_meta(bytes)?;
    let source = check(&meta, Some(expected_identity), ctx)?;
    let ChoiceResultSpec::Posterior { look } = source.spec() else {
        return Err(malformed("a retained choice source requires a pooled posterior result"));
    };
    let d = source.contract().declaration();
    let conflict_index = 4usize.saturating_mul(d.option_labels.len()).saturating_add(1);
    if source.result().values.get(conflict_index).is_none_or(|v| *v > 0.0) {
        return Err(malformed(
            "a conflicting posterior source cannot be reused until its conflict is resolved",
        ));
    }
    let respondents = source
        .contract()
        .respondents_at(*look as usize)
        .map_err(|r| Error::refused(r.code, r.detail, r.remedy.unwrap_or("")))?;
    let mut counts = vec![0.0; d.option_labels.len()];
    let mut observation_ids = Vec::new();
    for ancestor in &d.retained_sources {
        for (sum, count) in counts.iter_mut().zip(&ancestor.counts) {
            *sum += ancestor.discount * count;
        }
        observation_ids.extend(ancestor.observation_ids.iter().cloned());
    }
    for r in respondents {
        for (sum, count) in counts.iter_mut().zip(&r.counts) {
            *sum += f64::from(*count);
        }
    }
    Ok(RetainedChoiceSource {
        label: label.into(),
        option_labels: d.option_labels.clone(),
        meaning: d.meaning.clone(),
        population: d.population.clone(),
        counts,
        observation_ids: {
            observation_ids.extend(respondents.iter().map(|r| format!("{}#0", r.id)));
            observation_ids
        },
        discount,
        transfer_rationale: None,
        posterior_proof: Some(RetainedChoiceProof {
            artifact: bytes.to_vec(),
            expected_identity: expected_identity.into(),
        }),
    })
}

/// Replay every native posterior source and bind its frame, observations and initial prior to the target.
/// Declared raw observations remain ordinary input data and are not promoted to native posterior authority.
///
/// # Errors
/// A stale, altered, mismatched or otherwise unsupported retained posterior.
pub fn verify_retained_choice_sources(
    contract: &ChoiceDesignContract,
    ctx: &ExecutionContext,
) -> Result<(), Error> {
    thread_local! { static SOURCE_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
    struct DepthGuard;
    impl Drop for DepthGuard {
        fn drop(&mut self) {
            SOURCE_DEPTH.with(|d| d.set(d.get() - 1));
        }
    }
    let entered = SOURCE_DEPTH.with(|d| {
        let depth = d.get();
        if depth >= 8 {
            false
        } else {
            d.set(depth + 1);
            true
        }
    });
    if !entered {
        return Err(malformed("retained posterior ancestry exceeds eight levels"));
    }
    let _depth = DepthGuard;
    let target = contract.declaration();
    for source in &target.retained_sources {
        let Some(proof) = &source.posterior_proof else { continue };
        let quote = u64::try_from(proof.artifact.len())
            .ok()
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| malformed("retained posterior admission size overflow"))?;
        let _lease = ctx.acquire_admission(0, quote).map_err(|error| {
            if ctx.cancellation.is_cancelled() {
                Error::refused(
                    antecedent_core::reason_code!("execution_cancelled"),
                    "choice_inference.source_cancelled",
                    error.to_string(),
                )
            } else {
                Error::refused(
                    antecedent_core::reason_code!("execution_resource_limit"),
                    "choice_inference.source_resource_limit",
                    error.to_string(),
                )
            }
        })?;
        let meta = decode_meta(&proof.artifact)?;
        if reals(&meta.declaration.initial_prior_bits) != target.initial_prior {
            return Err(malformed("retained posterior and target must have the same base prior"));
        }
        let expected = retained_choice_source_from_artifact(
            &proof.artifact,
            &proof.expected_identity,
            &source.label,
            source.discount,
            ctx,
        )?;
        if &expected != source {
            return Err(malformed(
                "retained posterior source differs from its replayed observations or frame",
            ));
        }
    }
    Ok(())
}

fn encode_meta(meta: &ChoiceResultMeta, artifact_id: &str) -> Result<Vec<u8>, Error> {
    crate::artifact_codec::encode_sectioned(
        &crate::ArtifactContainer {
            kind: ARTIFACT_KIND,
            note: "choice_inference_result",
            max_bytes: MAX_CHOICE_INFERENCE_RESULT_BYTES,
            artifact_id,
            library_version: env!("CARGO_PKG_VERSION"),
        },
        META_SECTION,
        meta,
        &[],
    )
}

fn decode_meta(bytes: &[u8]) -> Result<ChoiceResultMeta, Error> {
    crate::artifact_codec::decode_sectioned(
        ARTIFACT_KIND,
        MAX_CHOICE_INFERENCE_RESULT_BYTES,
        META_SECTION,
        &[],
        CHOICE_INFERENCE_RESULT_ARTIFACT_VERSION,
        bytes,
    )
    .map(|(meta, _)| meta)
}

// -- public type -----------------------------------------------------------------------------

/// A validated result: the replayed choice contract, the request and the recomputed values.
#[derive(Clone, Debug)]
pub struct ChoiceInferenceResultArtifact {
    contract: ChoiceDesignContract,
    spec: ChoiceResultSpec,
    result: ChoiceResult,
    identity: String,
}

impl ChoiceInferenceResultArtifact {
    /// Compute a result and seal it with its contract.
    ///
    /// # Errors
    /// A refused authorization or request, or a numerical refusal.
    pub fn seal(
        contract: &ChoiceDesignContract,
        spec: &ChoiceResultSpec,
        ctx: &ExecutionContext,
    ) -> Result<(Self, ChoiceResultMeta), Error> {
        let result =
            compute_choice_result(contract, spec, ctx).map_err(|error| result_refusal(&error))?;
        let mut meta = ChoiceResultMeta {
            version: CHOICE_INFERENCE_RESULT_ARTIFACT_VERSION,
            feature: CHOICE_INFERENCE_RESULT_ARTIFACT_FEATURE.to_owned(),
            declaration: declaration_to_wire(contract.declaration()),
            contract_identity: contract.identity().to_hex(),
            spec: spec_to_wire(spec),
            method: result.method.to_owned(),
            value_bits: bits(&result.values),
            claim: "point_only".to_owned(),
            calibration: CHOICE_INFERENCE_CALIBRATION.to_owned(),
            closed_guarantee: result.closed_guarantee.to_owned(),
            identity: String::new(),
        };
        meta.identity = compute_identity(&meta);
        let artifact = check(&meta, None, ctx)?;
        Ok((artifact, meta))
    }

    /// The replayed contract.
    #[must_use]
    pub const fn contract(&self) -> &ChoiceDesignContract {
        &self.contract
    }

    /// The request.
    #[must_use]
    pub const fn spec(&self) -> &ChoiceResultSpec {
        &self.spec
    }

    /// The recomputed result.
    #[must_use]
    pub const fn result(&self) -> &ChoiceResult {
        &self.result
    }

    /// BLAKE3 identity (hex).
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Serialize a sealed result through the checksummed container.
    ///
    /// # Errors
    /// A result that does not seal, an encoding failure or an oversized payload.
    pub fn to_bytes(
        contract: &ChoiceDesignContract,
        spec: &ChoiceResultSpec,
        artifact_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, Error> {
        let (_, meta) = Self::seal(contract, spec, ctx)?;
        encode_meta(&meta, artifact_id)
    }

    /// Consume an artifact by replay and recomputation.
    ///
    /// # Errors
    /// Corruption, another major version, a contract refusal, a result that does not replay, or
    /// an identity that does not match the retained one.
    pub fn from_bytes(
        bytes: &[u8],
        expected_identity: Option<&str>,
        ctx: &ExecutionContext,
    ) -> Result<Self, Error> {
        check(&decode_meta(bytes)?, expected_identity, ctx)
    }

    /// Validate metadata as a consumer would, for tests that reseal edits.
    ///
    /// # Errors
    /// The same refusals as [`Self::from_bytes`].
    #[doc(hidden)]
    pub fn from_meta(
        meta: &ChoiceResultMeta,
        expected_identity: Option<&str>,
        ctx: &ExecutionContext,
    ) -> Result<Self, Error> {
        check(meta, expected_identity, ctx)
    }

    /// Recompute the identity of `meta` after an edit, for tests that reseal.
    #[doc(hidden)]
    pub fn reseal_identity(meta: &mut ChoiceResultMeta) {
        meta.identity = compute_identity(meta);
    }

    /// Decode stored metadata without validating it, for tests that edit it.
    ///
    /// # Errors
    /// Corruption, an unsupported layout or version.
    #[doc(hidden)]
    pub fn decode_unchecked(bytes: &[u8]) -> Result<ChoiceResultMeta, Error> {
        decode_meta(bytes)
    }
}
