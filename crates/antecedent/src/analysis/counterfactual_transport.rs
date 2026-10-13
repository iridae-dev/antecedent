//! Prepared lifecycle of general finite counterfactual transport.
//!
//! The stages are separate and each hands a typed object to the next:
//!
//! 1. [`prepare_counterfactual_transport`] decides the request once under one
//!    shared search budget: a derivation of a nested or conditional
//!    counterfactual probability of the target population from labelled
//!    observational and interventional laws, or a refusal. A proven
//!    obstruction (two finite models that agree on every regime of the declared
//!    evidence family and differ on the query) is returned as a
//!    [`CounterfactualTransportObstructionReport`]; every other non-identified
//!    outcome is a coded refusal that says which declared regime, selection
//!    target or limit would change it. Failure to derive is `not_certified`,
//!    never a proof of non-transportability.
//! 2. [`PreparedCounterfactualTransport::evaluate`] evaluates the stored
//!    derivation on supplied exact finite laws without re-identifying;
//!    [`PreparedCounterfactualTransport::refresh`] binds replaced evidence
//!    (changed numbers at the same shape included) and rebuilds the numerical
//!    state. A refresh that is refused leaves the prepared value untouched.
//! 3. [`CounterfactualTransportResult::export_artifact`] writes the
//!    `counterfactual_transport_v1` artifact and
//!    [`consume_counterfactual_transport_artifact`] replays it from bytes and
//!    an expected identity: it re-decides under the stored limits, requires the
//!    identical derivation, re-evaluates the point bit for bit, or re-verifies
//!    the obstruction's witness by exact enumeration.
//!
//! The route is point-only: exact finite laws carry no sampling uncertainty and
//! a counted, empirical or posterior law is refused with `cell_not_licensed`;
//! sampled counterfactual transport is a separate route.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{EvidenceCatalog, ExecutionContext, SearchLimits};
use antecedent_estimate::{
    CounterfactualTransportEvidence, CounterfactualTransportPoint,
    evaluate_counterfactual_transport,
};
use antecedent_expr::{ExactEvaluationLimits, ExactTransportData};
use antecedent_identify::{
    COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS, COUNTERFACTUAL_TRANSPORT_MEMORY_BYTES,
    CounterfactualTransportDecision, CounterfactualTransportDerivation,
    CounterfactualTransportObstruction, CounterfactualTransportProblem,
    CounterfactualTransportQuery, CounterfactualTransportWitnessCheck,
    counterfactual_transport_refusal, identify_counterfactual_transport,
};
use antecedent_io::IoError;
use antecedent_io::counterfactual_transport_wire::{
    CounterfactualTransportArtifactError, CounterfactualTransportExpectedIdentity,
};
use antecedent_io::counterfactual_transport_artifact::{
    ConsumedCounterfactualTransportOutcome, CounterfactualTransportConsumeLimits,
    consume_counterfactual_transport,
    identified_counterfactual_transport_artifact, obstructed_counterfactual_transport_artifact,
};

use super::transport_common::estimate_err;

/// What the caller asks of the decision beyond the query.
#[derive(Clone, Debug)]
pub struct CounterfactualTransportOptions {
    /// Search limits of the one decision (at most the contract's maxima).
    pub search: SearchLimits,
    /// Exact-evaluation operation and depth limits of every evaluation.
    pub evaluation: ExactEvaluationLimits,
    /// Name of every variable in ascending id order, or empty. Bound into the
    /// artifact's verified identity so a consumer can refuse relabelled names.
    pub variable_names: Vec<String>,
}

impl Default for CounterfactualTransportOptions {
    fn default() -> Self {
        Self {
            search: COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS,
            evaluation: ExactEvaluationLimits::default(),
            variable_names: Vec::new(),
        }
    }
}

/// The decision of one request.
#[derive(Clone, Debug)]
pub enum CounterfactualTransportPreparation {
    /// A checked derivation, ready to evaluate on supplied laws.
    Identified(Box<PreparedCounterfactualTransport>),
    /// A verified two-model witness that the declared evidence does not
    /// determine the query.
    ProvenNonTransportable(Box<CounterfactualTransportObstructionReport>),
}

impl CounterfactualTransportPreparation {
    /// The prepared derivation, when identified.
    #[must_use]
    pub fn identified(&self) -> Option<&PreparedCounterfactualTransport> {
        match self {
            Self::Identified(prepared) => Some(prepared),
            Self::ProvenNonTransportable(_) => None,
        }
    }

    /// The obstruction, when proven non-transportable.
    #[must_use]
    pub fn obstruction(&self) -> Option<&CounterfactualTransportObstructionReport> {
        match self {
            Self::ProvenNonTransportable(report) => Some(report),
            Self::Identified(_) => None,
        }
    }
}

/// A proven obstruction with the limits it was decided under.
#[derive(Clone, Debug)]
pub struct CounterfactualTransportObstructionReport {
    obstruction: CounterfactualTransportObstruction,
    search: SearchLimits,
    memory_limit_bytes: u64,
    variable_names: Vec<String>,
}

impl CounterfactualTransportObstructionReport {
    /// The obstruction: model class, query, catalog and portable witness.
    #[must_use]
    pub const fn obstruction(&self) -> &CounterfactualTransportObstruction {
        &self.obstruction
    }

    /// Why no derivation exists (descriptive).
    #[must_use]
    pub fn reason(&self) -> &str {
        &self.obstruction.reason
    }

    /// Re-verify the witness by exact rational enumeration on the report's own
    /// model class, query and catalog.
    ///
    /// # Errors
    /// `transport_not_certified` (`counterfactual_transport.invalid_witness`)
    /// when the witness no longer verifies.
    pub fn verify(&self) -> Result<CounterfactualTransportWitnessCheck, IoError> {
        self.obstruction.verify().map_err(identification_error)
    }

    /// Export the obstruction as a `counterfactual_transport_v1` artifact that a
    /// fresh consumer re-verifies from bytes.
    ///
    /// # Errors
    /// The artifact does not encode or names do not cover the model class.
    pub fn export_artifact(&self) -> Result<Vec<u8>, IoError> {
        obstructed_counterfactual_transport_artifact(
            &self.obstruction,
            self.search,
            self.memory_limit_bytes,
            &self.variable_names,
        )
        .and_then(|wire| wire.export())
        .map_err(artifact_error)
    }
}

/// The point of one evaluation with everything needed to export it.
#[derive(Clone, Debug)]
pub struct CounterfactualTransportResult {
    point: CounterfactualTransportPoint,
    derivation: CounterfactualTransportDerivation,
    data: ExactTransportData,
    evaluation: ExactEvaluationLimits,
    variable_names: Vec<String>,
}

impl CounterfactualTransportResult {
    /// The requested probability `P*(event | conditioning)`.
    #[must_use]
    pub const fn probability(&self) -> f64 {
        self.point.probability
    }

    /// `P*(event and conditioning)` of the (possibly reduced) query.
    #[must_use]
    pub const fn numerator(&self) -> f64 {
        self.point.numerator
    }

    /// `P*(conditioning)` of the (possibly reduced) query; one when unconditional.
    #[must_use]
    pub const fn denominator(&self) -> f64 {
        self.point.denominator
    }

    /// The original conditioning event's mass, when the derivation carries it.
    #[must_use]
    pub const fn conditioning_mass(&self) -> Option<f64> {
        self.point.conditioning_mass
    }

    /// Whether the original conditioning event was checked to have positive
    /// mass; `false` means the value holds conditional on that premise.
    #[must_use]
    pub const fn conditioning_mass_checked(&self) -> bool {
        self.point.conditioning_mass_checked
    }

    /// Regime-law cells read.
    #[must_use]
    pub const fn cells_read(&self) -> u64 {
        self.point.cells_read
    }

    /// Every law the evaluation read, with its population, regime, snapshot and
    /// content identity.
    #[must_use]
    pub fn cited_laws(&self) -> &[CounterfactualTransportEvidence] {
        &self.point.evidence
    }

    /// The full point.
    #[must_use]
    pub const fn point(&self) -> &CounterfactualTransportPoint {
        &self.point
    }

    /// The derivation the point was evaluated from.
    #[must_use]
    pub const fn derivation(&self) -> &CounterfactualTransportDerivation {
        &self.derivation
    }

    /// The exact laws the point was computed on.
    #[must_use]
    pub const fn data(&self) -> &ExactTransportData {
        &self.data
    }

    /// Export the premises, derivation, laws and point as a
    /// `counterfactual_transport_v1` artifact.
    ///
    /// # Errors
    /// The artifact does not encode, a bound is exceeded or names do not cover
    /// the model class.
    pub fn export_artifact(&self) -> Result<Vec<u8>, IoError> {
        identified_counterfactual_transport_artifact(
            &self.derivation,
            &self.data,
            self.evaluation,
            &self.variable_names,
            &self.point,
        )
        .and_then(|wire| wire.export())
        .map_err(artifact_error)
    }

    /// The identity a consumer should hold: the artifact's premises and data
    /// digests, independent of the bytes.
    ///
    /// # Errors
    /// The artifact does not encode.
    pub fn identity(&self) -> Result<CounterfactualTransportExpectedIdentity, IoError> {
        identified_counterfactual_transport_artifact(
            &self.derivation,
            &self.data,
            self.evaluation,
            &self.variable_names,
            &self.point,
        )
        .map(|wire| wire.identity())
        .map_err(artifact_error)
    }
}

/// A decided query: the frozen derivation, its limits and, once evidence is
/// bound, the numerical state built from it.
#[derive(Clone, Debug)]
pub struct PreparedCounterfactualTransport {
    derivation: CounterfactualTransportDerivation,
    evaluation: ExactEvaluationLimits,
    variable_names: Vec<String>,
    bound: Option<CounterfactualTransportResult>,
}

fn identification_error(error: antecedent_identify::IdentificationError) -> IoError {
    match counterfactual_transport_refusal(&error) {
        Some((code, detail)) => IoError::Refused { code, message: format!("{detail}: {error}") },
        None => IoError::from(error),
    }
}

fn artifact_error(error: CounterfactualTransportArtifactError) -> IoError {
    let (code, message) = error.refusal();
    IoError::Refused { code, message }
}

/// Describe a decision that is not an identification as the coded refusal that
/// names what would change it.
fn refusal_of(decision: &CounterfactualTransportDecision) -> IoError {
    let code = decision
        .reason_code()
        .unwrap_or(antecedent_core::reason_code!("transport_not_certified"));
    let detail = decision.detail_code().unwrap_or_default();
    let text = match decision {
        CounterfactualTransportDecision::MissingEvidence(missing) => {
            let regimes: Vec<String> = missing
                .missing
                .iter()
                .map(|m| {
                    let vars: Vec<String> =
                        m.interventions.iter().map(|v| v.raw().to_string()).collect();
                    if vars.is_empty() {
                        format!("the observational law of population {}", m.population)
                    } else {
                        format!(
                            "the experiment on [{}] in population {}",
                            vars.join(", "),
                            m.population
                        )
                    }
                })
                .collect();
            format!(
                "{}; supply a catalog regime and an exact law for {}",
                missing.detail,
                regimes.join("; ")
            )
        }
        CounterfactualTransportDecision::UnsupportedScope(unsupported) => {
            unsupported.message.clone()
        }
        CounterfactualTransportDecision::NotCertified(inspection) => format!(
            "the bounded search found neither a derivation nor a verified two-model witness, \
             which is not a proof of non-transportability ({}; {})",
            inspection.failures.join("; "),
            inspection.witness
        ),
        CounterfactualTransportDecision::Exhausted(receipt) => {
            format!("the search stopped before deciding: {}", receipt.summary())
        }
        CounterfactualTransportDecision::ProvenNonTransportable(obstruction) => {
            obstruction.reason.clone()
        }
        CounterfactualTransportDecision::Identified(_) => String::new(),
    };
    IoError::Refused { code, message: format!("{detail}: {text}") }
}

/// Decide a counterfactual transport request once.
///
/// `evidence`, when supplied, is validated against the derivation's catalog and
/// evaluated immediately (the numerical state); without it, evaluate or refresh
/// later. Nothing is searched again after this call.
///
/// # Errors
/// A coded refusal: `route_not_supported` for a request outside the frozen
/// contract, `invalid_argument` for a malformed query or catalog,
/// `transport_missing_evidence` naming the regimes that would license a
/// derivation, `transport_not_certified` when the bounded search neither
/// derives nor verifies a witness, `transport_budget_cancel` with the search
/// receipt, and the evaluation refusals of [`PreparedCounterfactualTransport::evaluate`]
/// for evidence that is supplied.
pub fn prepare_counterfactual_transport(
    problem: &CounterfactualTransportProblem,
    query: &CounterfactualTransportQuery,
    catalog: &EvidenceCatalog,
    evidence: Option<ExactTransportData>,
    options: &CounterfactualTransportOptions,
    ctx: &ExecutionContext,
) -> Result<CounterfactualTransportPreparation, IoError> {
    let names = &options.variable_names;
    if !names.is_empty() && names.len() != problem.variable_count() {
        return Err(IoError::Refused {
            code: antecedent_core::reason_code!("invalid_argument"),
            message: format!(
                "counterfactual_transport.invalid_query: {} variable names for {} variables",
                names.len(),
                problem.variable_count()
            ),
        });
    }
    let decision =
        identify_counterfactual_transport(problem, query, catalog, options.search, ctx)
            .map_err(identification_error)?;
    match decision {
        CounterfactualTransportDecision::Identified(derivation) => {
            let prepared = PreparedCounterfactualTransport {
                derivation: *derivation,
                evaluation: options.evaluation,
                variable_names: options.variable_names.clone(),
                bound: None,
            };
            let prepared = match evidence {
                Some(data) => prepared.refresh(data, ctx)?,
                None => prepared,
            };
            Ok(CounterfactualTransportPreparation::Identified(Box::new(prepared)))
        }
        CounterfactualTransportDecision::ProvenNonTransportable(obstruction) => {
            let memory_limit_bytes = ctx.memory.hard_limit_bytes.map_or(
                COUNTERFACTUAL_TRANSPORT_MEMORY_BYTES,
                |hard| hard.min(COUNTERFACTUAL_TRANSPORT_MEMORY_BYTES),
            );
            Ok(CounterfactualTransportPreparation::ProvenNonTransportable(Box::new(
                CounterfactualTransportObstructionReport {
                    obstruction: *obstruction,
                    search: options.search,
                    memory_limit_bytes,
                    variable_names: options.variable_names.clone(),
                },
            )))
        }
        other => Err(refusal_of(&other)),
    }
}

impl PreparedCounterfactualTransport {
    /// The frozen derivation.
    #[must_use]
    pub const fn derivation(&self) -> &CounterfactualTransportDerivation {
        &self.derivation
    }

    /// The model class the derivation was decided in.
    #[must_use]
    pub const fn problem(&self) -> &CounterfactualTransportProblem {
        &self.derivation.problem
    }

    /// The canonical query.
    #[must_use]
    pub const fn query(&self) -> &CounterfactualTransportQuery {
        &self.derivation.query
    }

    /// The bound laws, when evidence has been bound.
    #[must_use]
    pub fn data(&self) -> Option<&ExactTransportData> {
        self.bound.as_ref().map(CounterfactualTransportResult::data)
    }

    /// Evaluate the frozen derivation on `evidence` without re-identifying and
    /// without changing this value.
    ///
    /// # Errors
    /// `cell_not_licensed` (`counterfactual_transport.sampled_inference_closed`)
    /// for a counted, empirical or posterior law; `invalid_argument`
    /// (`counterfactual_transport.invalid_evidence`) for a law whose population,
    /// regime, axes or levels do not match the catalog and declared levels;
    /// `transport_missing_provider` (`counterfactual_transport.missing_law`)
    /// when a cited regime has no law at a needed intervention level;
    /// `transport_support_failure` for an undefined factor or a conditioning
    /// event of zero mass; `transport_numerical_failure` when the laws contradict
    /// the graph; `transport_budget_cancel` for a limit or cancellation.
    pub fn evaluate(
        &self,
        evidence: &ExactTransportData,
        ctx: &ExecutionContext,
    ) -> Result<CounterfactualTransportResult, IoError> {
        antecedent_estimate::refuse_cancelled(ctx, "counterfactual-transport evaluation")
            .map_err(estimate_err)?;
        let point = evaluate_counterfactual_transport(
            &self.derivation,
            evidence,
            self.evaluation,
            ctx,
        )
        .map_err(estimate_err)?;
        Ok(CounterfactualTransportResult {
            point,
            derivation: self.derivation.clone(),
            data: evidence.clone(),
            evaluation: self.evaluation,
            variable_names: self.variable_names.clone(),
        })
    }

    /// The numerical state built from the bound evidence.
    ///
    /// # Errors
    /// `transport_missing_provider` when no evidence has been bound;
    /// `transport_budget_cancel` after cancellation.
    pub fn estimate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<&CounterfactualTransportResult, IoError> {
        antecedent_estimate::refuse_cancelled(ctx, "counterfactual-transport estimate")
            .map_err(estimate_err)?;
        self.bound.as_ref().ok_or_else(|| IoError::Refused {
            code: antecedent_core::reason_code!("transport_missing_provider"),
            message: "counterfactual_transport.missing_law: no evidence is bound; call refresh \
                      with exact laws for the regimes the derivation cites or evaluate on \
                      supplied evidence"
                .into(),
        })
    }

    /// Bind `evidence` and rebuild the numerical state. The derivation, its
    /// search receipt and the evaluation limits are unchanged; laws of another
    /// population or regime are refused, which forces a new preparation. Changed
    /// numbers at an unchanged shape replace the previous numbers: nothing
    /// numerical survives a refresh.
    ///
    /// # Errors
    /// As [`Self::evaluate`]; this value is untouched on error.
    pub fn refresh(
        &self,
        evidence: ExactTransportData,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        let result = self.evaluate(&evidence, ctx)?;
        Ok(Self { bound: Some(result), ..self.clone() })
    }
}

/// The conclusion of a consumed artifact.
#[derive(Clone, Debug)]
pub enum CounterfactualTransportConsumption {
    /// An identified query with its recomputed point.
    Identified(Box<CounterfactualTransportResult>),
    /// A proven obstruction with its re-verified witness.
    ProvenNonTransportable(Box<CounterfactualTransportObstructionReport>),
}

/// Independently consume a counterfactual transport artifact: decode, check
/// every digest and the expected identity, re-decide under the stored limits
/// (refusing limits above `limits` before any work), require the identical
/// derivation and a bit-for-bit point, or re-verify the obstruction's witness.
/// No data or provider is fetched.
///
/// # Errors
/// [`IoError::Refused`] with the failure's registered code and a
/// `counterfactual_transport.*` detail naming the failed slot. Use
/// [`antecedent_io::counterfactual_transport_artifact::consume_counterfactual_transport_with_limits`]
/// for the typed error.
pub fn consume_counterfactual_transport_artifact(
    bytes: &[u8],
    expected: &CounterfactualTransportExpectedIdentity,
    limits: CounterfactualTransportConsumeLimits,
    ctx: &ExecutionContext,
) -> Result<CounterfactualTransportConsumption, IoError> {
    let consumed = consume_counterfactual_transport(bytes, expected, limits, ctx)?;
    let variable_names = consumed.wire.variable_names.clone();
    Ok(match consumed.outcome {
        ConsumedCounterfactualTransportOutcome::Identified(identified) => {
            let identified = *identified;
            CounterfactualTransportConsumption::Identified(Box::new(
                CounterfactualTransportResult {
                    point: identified.point,
                    derivation: identified.derivation,
                    data: identified.data,
                    evaluation: ExactEvaluationLimits {
                        operations: consumed.wire.operation_limit,
                        depth: consumed.wire.depth_limit,
                    },
                    variable_names,
                },
            ))
        }
        ConsumedCounterfactualTransportOutcome::Obstructed(obstructed) => {
            let obstructed = *obstructed;
            CounterfactualTransportConsumption::ProvenNonTransportable(Box::new(
                CounterfactualTransportObstructionReport {
                    obstruction: CounterfactualTransportObstruction {
                        query: obstructed.query,
                        problem: obstructed.problem,
                        catalog: obstructed.catalog,
                        witness: obstructed.witness,
                        check: obstructed.check,
                        reason: obstructed.reason,
                    },
                    search: SearchLimits {
                        operations: consumed.wire.search.operations,
                        depth: consumed.wire.search.depth,
                    },
                    memory_limit_bytes: consumed.wire.search.memory_limit_bytes,
                    variable_names,
                },
            ))
        }
    })
}
