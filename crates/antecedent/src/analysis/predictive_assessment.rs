//! Predictive assessment as a facade object: score predictive laws against
//! realized observations under a declared design, with the overlap check taken
//! from the evidence ledger.
//!
//! The scoring contract lives in [`antecedent_validate::predictive_assessment`];
//! this module routes requests through it and adds no scientific rule of its
//! own:
//!
//! * **Overlap from the ledger.** [`fit_declaration_from_ledger`] resolves what a
//!   law was fitted or tuned on from the evidence ledger's row-level observation
//!   identities (derivation lineage already applied), and
//!   [`observation_keys_from_ledger`] resolves the assessed observations the
//!   same way, so held-out is decided from identities and never from a label.
//!   An undeclared source gives an `unverified` assessment, never a held-out one.
//! * **One scoring surface.** [`posterior_predictive_predictor`] turns a
//!   Bayesian posterior predictive check's replicates into sample laws scored
//!   by the same CRPS and PIT as any other law.
//! * **Point values and a record-gated interval.** Scores, paired differences and
//!   diagnostics are point values; the interval of an expected score or score
//!   difference ([`PreparedPredictiveAssessment::expected_score_interval`]) is published only
//!   inside a measured design and an unscoped request
//!   ([`PreparedPredictiveAssessment::request_score_interval`]) refuses, a
//!   predictive score is not causal validity
//!   ([`PreparedPredictiveAssessment::request_causal_validity`]) and not the
//!   probability that a claim holds
//!   ([`PreparedPredictiveAssessment::request_score_as_probability`]).
//!
//! # Artifact
//!
//! [`PreparedPredictiveAssessment::export`] writes `predictive_assessment_v1`;
//! [`consume_predictive_assessment_artifact`] re-scores every case from the
//! stored laws and observations and refuses a stored identity or value that does
//! not replay, including a consistently resealed edit when the consumer retained
//! the identity independently.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::evidence_ledger::{
    EffectiveObservations, EvidenceLedger, ObservationId, SourceId,
};
use antecedent_core::ExecutionContext;
use antecedent_estimate::{CausalPosterior, PreparedBayesianProblem};
use antecedent_io::IoError;
use antecedent_io::predictive_assessment_artifact::{
    PredictiveAssessmentArtifact, PredictiveAssessmentArtifactError,
};
use antecedent_validate::PosteriorPredictiveCheck;
use antecedent_validate::predictive_assessment::PredictiveAssessmentError;

pub use antecedent_io::predictive_assessment_artifact::{
    PREDICTIVE_ASSESSMENT_ARTIFACT_FEATURE, PREDICTIVE_ASSESSMENT_ARTIFACT_VERSION,
    PREDICTIVE_ASSESSMENT_CAVEATS, PredictiveAssessmentConsumeLimits,
};
pub use antecedent_validate::predictive_assessment::{
    AssessmentIdentity, DependenceDeclaration, DistributionDrift, EnsembleCrpsForm, FiniteTruth,
    ForecastLaw, Horizon, LawSource, Outcome, OverlapPolicy, OverlapStatus, PredictiveReplicates,
    ProviderIdentity, ProviderTrustLabel, RealizedObservations, ScoreDifference, ScoreInterval,
    ScoreIntervalQuery, ScoreIntervalRequest, ScoreKind, ValidationDesign, distribution_drift,
    exact_expected_score,
};
pub use antecedent_validate::predictive_assessment::{
    FitDeclaration, ObservationKey, PredictiveAssessmentReport, PredictiveAssessmentSpec,
    PredictiveTarget, PredictorLaws,
};

/// One refusal as plain data: registered reason code, namespaced
/// `predictive_assessment.*` detail and the explanation of what was refused and
/// what would license it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PredictiveAssessmentRefusal {
    /// Registered runtime reason code.
    pub code: &'static str,
    /// Namespaced detail, `predictive_assessment.<rule>`.
    pub detail: String,
    /// What was refused and what would license it.
    pub message: String,
}

impl PredictiveAssessmentRefusal {
    fn new(code: &'static str, detail: &str, message: impl Into<String>) -> Self {
        Self { code, detail: detail.to_owned(), message: message.into() }
    }

    fn from_contract(error: &PredictiveAssessmentError) -> Self {
        Self::new(error.reason_code(), error.detail(), error.message.clone())
    }

    fn from_artifact(error: &PredictiveAssessmentArtifactError) -> Self {
        match error.refusal() {
            Some((code, detail, message)) => Self::new(code, &detail, message),
            None => Self::new(
                antecedent_core::reason_code!("invalid_argument"),
                "predictive_assessment.invalid_artifact",
                error.to_string(),
            ),
        }
    }
}

impl std::fmt::Display for PredictiveAssessmentRefusal {
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

impl std::error::Error for PredictiveAssessmentRefusal {}

impl From<PredictiveAssessmentRefusal> for IoError {
    fn from(refusal: PredictiveAssessmentRefusal) -> Self {
        Self::Refused {
            code: refusal.code,
            message: format!("{}: {}", refusal.detail, refusal.message),
        }
    }
}

impl From<PredictiveAssessmentRefusal> for crate::CausalError {
    fn from(refusal: PredictiveAssessmentRefusal) -> Self {
        Self::Serialization(refusal.into())
    }
}

fn key_of(id: ObservationId) -> ObservationKey {
    ObservationKey::from_bytes(*id.digest().as_bytes())
}

/// What a law was fitted or tuned on, from the ledger's resolved observation
/// identities of the sources it was fitted and tuned on.
///
/// The union of the sources' effective observations (lineage applied) is the fit
/// set. A source whose observations are not computable, or no source at all,
/// gives [`FitDeclaration::Undeclared`]: the assessment is then `unverified`,
/// never held out.
///
/// # Errors
/// A source id that is not in the ledger.
pub fn fit_declaration_from_ledger(
    ledger: &EvidenceLedger,
    fit_sources: &[SourceId],
) -> Result<FitDeclaration, PredictiveAssessmentRefusal> {
    if fit_sources.is_empty() {
        return Ok(FitDeclaration::Undeclared {
            reason: "no fit source was declared in the ledger".to_owned(),
        });
    }
    let mut keys: Vec<ObservationKey> = Vec::new();
    for id in fit_sources {
        let Some(source) = ledger.source(*id) else {
            return Err(PredictiveAssessmentRefusal::new(
                antecedent_core::reason_code!("invalid_argument"),
                "predictive_assessment.unknown_source",
                format!("fit source {id} is not in the evidence ledger"),
            ));
        };
        match &source.effective {
            EffectiveObservations::Known(rows) => keys.extend(rows.iter().copied().map(key_of)),
            EffectiveObservations::Unknown(reason) => {
                return Ok(FitDeclaration::Undeclared {
                    reason: format!(
                        "the observations behind fit source `{}` are not declared: {reason}",
                        source.declaration.label
                    ),
                });
            }
        }
    }
    keys.sort_unstable();
    keys.dedup();
    Ok(FitDeclaration::FittedOn(keys))
}

/// The assessed observations' identities from one ledger observation source.
///
/// # Errors
/// An unknown source, or one whose observations are not declared: held-out
/// cannot be decided without identities.
pub fn observation_keys_from_ledger(
    ledger: &EvidenceLedger,
    source: SourceId,
) -> Result<Vec<ObservationKey>, PredictiveAssessmentRefusal> {
    let Some(entry) = ledger.source(source) else {
        return Err(PredictiveAssessmentRefusal::new(
            antecedent_core::reason_code!("invalid_argument"),
            "predictive_assessment.unknown_source",
            format!("observation source {source} is not in the evidence ledger"),
        ));
    };
    match &entry.effective {
        EffectiveObservations::Known(rows) => Ok(rows.iter().copied().map(key_of).collect()),
        EffectiveObservations::Unknown(reason) => Err(PredictiveAssessmentRefusal::new(
            antecedent_core::reason_code!("evidence_overlap_unknown"),
            "predictive_assessment.overlap_unknown",
            format!(
                "the observations of source `{}` are not declared ({reason}); declare their \
                 row-level identities in the ledger so that held-out can be decided",
                entry.declaration.label
            ),
        )),
    }
}

/// The posterior predictive replicates of a Bayesian check as a named predictor
/// of the observed rows, with the observed outcomes as realized outcomes.
///
/// Each row's replicates become a sample law scored by the CRPS and checked by
/// the PIT through the one scoring surface; the check's own discrepancy
/// statistics and tail probabilities stay on its report. The replicates are
/// drawn at parameters fitted to these rows, so `fit` should name the rows the
/// posterior was conditioned on.
///
/// # Errors
/// Missing coefficients or posterior draws.
pub fn posterior_predictive_predictor(
    check: &PosteriorPredictiveCheck,
    problem: &PreparedBayesianProblem,
    posterior: &CausalPosterior,
    name: &str,
    model: &str,
    target: PredictiveTarget,
    fit: FitDeclaration,
) -> Result<(PredictorLaws, Vec<Outcome>), PredictiveAssessmentRefusal> {
    let replicates = check.replicates(problem, posterior).map_err(|error| {
        PredictiveAssessmentRefusal::new(
            antecedent_core::reason_code!("invalid_argument"),
            "predictive_assessment.invalid_law",
            error.to_string(),
        )
    })?;
    let outcomes = problem.design.outcome.iter().map(|y| Outcome::Real(*y)).collect();
    Ok((replicates.into_predictor(name, model, target, fit), outcomes))
}

/// A sealed predictive assessment: the report, its identity and its artifact.
#[derive(Clone, Debug)]
pub struct PreparedPredictiveAssessment {
    artifact: PredictiveAssessmentArtifact,
}

impl PreparedPredictiveAssessment {
    /// Assess the declared laws against the realized observations.
    ///
    /// # Errors
    /// Any refusal of the assessment contract: a target or horizon mismatch, a
    /// malformed law, a callback law, an unsupported score, an iid declaration
    /// for a temporal or clustered design, an unmet held-out requirement, a
    /// requested interval, an exceeded bound or a cancellation.
    pub fn assess(
        spec: &PredictiveAssessmentSpec,
        ctx: &ExecutionContext,
    ) -> Result<Self, PredictiveAssessmentRefusal> {
        PredictiveAssessmentArtifact::seal(spec, ctx)
            .map(|artifact| Self { artifact })
            .map_err(|error| PredictiveAssessmentRefusal::from_artifact(&error))
    }

    /// The scores, paired differences and diagnostics.
    #[must_use]
    pub fn report(&self) -> &PredictiveAssessmentReport {
        self.artifact.report()
    }

    /// The identity digests of the inputs.
    #[must_use]
    pub fn identity(&self) -> &AssessmentIdentity {
        self.artifact.identity()
    }

    /// The assessment as declared, in canonical case order.
    #[must_use]
    pub fn spec(&self) -> &PredictiveAssessmentSpec {
        self.artifact.spec()
    }

    /// Serialize the portable `predictive_assessment_v1` artifact.
    ///
    /// # Errors
    /// An empty id, an encoding failure or an oversized payload.
    pub fn export(&self, artifact_id: &str) -> Result<Vec<u8>, PredictiveAssessmentRefusal> {
        self.artifact
            .to_bytes(artifact_id)
            .map_err(|error| PredictiveAssessmentRefusal::from_artifact(&error))
    }

    /// The interval of one predictor's expected score, or of the expected difference of
    /// two predictors' scores, under the declared dependence.
    ///
    /// Published only for a held-out or unfitted predictor where a current attesting
    /// coverage record measured the construction at this design, score, law family,
    /// level and number of cases; the record is resolved from the shipped coverage table.
    ///
    /// # Errors
    /// `cell_not_licensed` outside a measured design, an undeclared dependence, an
    /// in-sample or unverified predictor, or a degenerate sample variance.
    pub fn expected_score_interval(
        &self,
        query: &ScoreIntervalQuery,
    ) -> Result<ScoreInterval, PredictiveAssessmentRefusal> {
        antecedent_validate::predictive_assessment::expected_score_interval(
            self.spec(),
            self.report(),
            query,
        )
        .map_err(|error| PredictiveAssessmentRefusal::from_contract(&error))
    }

    /// Refuse an unscoped score or score-difference interval request (no predictor,
    /// score or level named); ask [`Self::expected_score_interval`] for a scoped one.
    ///
    /// # Errors
    /// Always `cell_not_licensed`.
    pub fn request_score_interval(
        &self,
    ) -> Result<std::convert::Infallible, PredictiveAssessmentRefusal> {
        self.report()
            .request_score_interval()
            .map_err(|error| PredictiveAssessmentRefusal::from_contract(&error))
    }

    /// Refuse reading a predictive score as causal validity.
    ///
    /// # Errors
    /// Always `route_not_supported`.
    pub fn request_causal_validity(
        &self,
    ) -> Result<std::convert::Infallible, PredictiveAssessmentRefusal> {
        self.report()
            .request_causal_validity()
            .map_err(|error| PredictiveAssessmentRefusal::from_contract(&error))
    }

    /// Refuse reading a model score as the probability that a claim holds.
    ///
    /// # Errors
    /// Always `route_not_supported`.
    pub fn request_score_as_probability(
        &self,
    ) -> Result<std::convert::Infallible, PredictiveAssessmentRefusal> {
        self.report()
            .request_score_as_probability()
            .map_err(|error| PredictiveAssessmentRefusal::from_contract(&error))
    }
}

/// Consume a `predictive_assessment_v1` artifact by re-scoring every case.
///
/// `expected` is an identity the caller retained independently of the bytes: a
/// consistently resealed artifact of other laws or observations is refused.
///
/// # Errors
/// Corruption, another version, unsupported claims, a bound exceeded, a changed
/// identity or a stored value that does not replay.
pub fn consume_predictive_assessment_artifact(
    bytes: &[u8],
    expected: Option<&AssessmentIdentity>,
    limits: PredictiveAssessmentConsumeLimits,
    ctx: &ExecutionContext,
) -> Result<PreparedPredictiveAssessment, PredictiveAssessmentRefusal> {
    PredictiveAssessmentArtifact::from_bytes_with_limits(bytes, expected, limits, ctx)
        .map(|artifact| PreparedPredictiveAssessment { artifact })
        .map_err(|error| PredictiveAssessmentRefusal::from_artifact(&error))
}
