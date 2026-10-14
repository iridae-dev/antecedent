//! Licensed forward sources of an inverse query.
//!
//! An inverse query asks which declared actions meet a target on a forward
//! claim. The claim is only as good as the route that produced it, so each
//! source the query may read is adapted here with its own refusals and with its
//! identity carried into the result:
//!
//! - **Composed mechanisms** ([`composed_forward_source`]): the replayed result
//!   of a mechanism-composition query per action. A linear-Gaussian composition
//!   supplies the interventional mean of each action (a point); a finite-table
//!   composition supplies the exact finite law of each action, given the
//!   caller's numeric coding of the outcome levels. A Gaussian composition does
//!   not supply a law the engine can integrate, so a quantile or a probability
//!   on it is refused with [`ForwardSourceRefusal::ComposedLawNotMaterialized`]
//!   and the remedy of drawing the intervened law. A composition whose
//!   mechanisms were carried across populations is a transported claim: it is
//!   refused under `NativeTargetOnly` and, under `AcceptsTransported`
//!   ([`composed_forward_source_accepting`]), carries the transport receipt and
//!   the receipt's support label instead of `supported`.
//! - **Cohort-mean contrast** ([`cohort_forward_source`]): the weighted mean
//!   contrast of a retained effect model over a declared cohort. A point. A
//!   cohort carried under a declared covariate shift is a transported
//!   response and is read only when the consumer accepts transported claims.
//! - **Transported responses** ([`transported_forward_source`]): a value
//!   carried to a target population (cohort effect, policy value, synthesis,
//!   mechanism transport), as a point or as a transported law.
//!
//! # Refusals
//!
//! Each source states what it cannot supply before the engine runs.
//! A point (every composed Gaussian value, every cohort contrast, every
//! transported mean) is refused a quantile, a probability, a nonlinear utility
//! and an outcome-law input ([`ForwardSourceRefusal::PointOnlyLawRequested`]).
//! No source supplies an interval region: it is refused for every source
//! ([`ForwardSourceRefusal::IntervalNotLicensed`]), because the composition
//! variance, the cohort model's standard error and the transport step have no
//! measured coverage. A source's own variance is retained as a diagnostic and
//! never becomes a feasibility statement.
//!
//! # Identity
//!
//! [`ForwardOrigin`] records the kind of source, its identity, its
//! representation, its support label, the assumed mechanisms (composition) or
//! the whole [`TransportReceipt`] (transport) and a digest of the claim that
//! was handed to the engine. [`SourcedInverseResult`] returns it beside the
//! result; [`seal_on_source`] binds it into the inverse-query artifact so a
//! consumer re-derives the claim digest and refuses an origin that does not
//! describe the embedded claim.
//!
//! The origin is a declaration of where the numbers came from. Replaying an
//! artifact checks that the embedded numbers are the numbers the origin names;
//! it does not re-execute the producing route.
//!
//! # Ownership
//!
//! Sources own their claims. Evaluation borrows the query and the source and
//! retains neither.
//!
//! # Refresh
//!
//! Refresh with outcome-only data. A forward source owns one claim and the
//! origin that produced it; a refreshed composition, cohort model or transported
//! value is a new source with a new claim digest and a new origin identity, and
//! an inverse query is re-evaluated on it (nothing is refreshed in place). The
//! query, the declared actions and the consumer's stance on transported claims
//! are kept; the claim, its support label and its transport receipt are
//! recomputed. A refresh that would carry mechanisms across populations changes
//! the source from native to transported and is refused under
//! `NativeTargetOnly`; data of another population, quantity or arm is refused
//! by the producing route before it reaches here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::provider_transport::TransportedInputPolicy;
use antecedent_contract::{ExternalRefusal, ScientificQuantity};
use antecedent_core::{CanonicalWriter, SupportStatus};
use antecedent_io_core::cohort_decision::CohortDecisionSource;
use antecedent_io_core::convert::to_cbor;
use antecedent_io_core::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io_core::error::IoError;
use antecedent_io_core::mechanism_composition_artifact::{
    CompositionQueryWire, CompositionResultWire,
};
use antecedent_io_core::quantity_wire::DistributionMeaningWire;
use serde::{Deserialize, Serialize};

use crate::cohort_decision_input::cohort_mean_source;
use crate::decision_contract::DecisionContract;
use crate::decision_eval::MeanSource;
use crate::decision_transported_source::{
    TransportReceipt, TransportedRefusal, TransportedRepresentation, TransportedResponse,
    TransportedRoute, TransportedValue,
};
use crate::inverse_query::{
    ForwardClaim, ForwardEvidence, InverseConstraint, InverseQuery, InverseQueryError,
    InverseResult, evaluate_inverse_query, evaluate_inverse_query_per_action,
};
use crate::inverse_query_artifact::InverseQueryArtifact;
use crate::signal::RefusalBuilder;

/// Version of the origin record and its identity encoding.
pub const FORWARD_ORIGIN_VERSION: u16 = 1;

const STAGE: &str = "inverse_forward_source";
const TAG_CLAIM: &str = "antecedent.inverse_forward_source.claim";
const TAG_ORIGIN: &str = "antecedent.inverse_forward_source.origin";
const TAG_COMPOSED: &str = "antecedent.inverse_forward_source.composed";
const PROBABILITY_SUM_TOLERANCE: f64 = 1e-9;

/// Detail: a transported response is offered where only native target data is accepted.
pub const DETAIL_TRANSPORTED_NOT_ACCEPTED: &str = "inverse_forward_source.transported_not_accepted";
/// Detail: a composed response is malformed.
pub const DETAIL_COMPOSED_INVALID: &str = "inverse_forward_source.composed_response_invalid";
/// Detail: a composed query and its stored result are different kinds.
pub const DETAIL_COMPOSED_MISMATCH: &str = "inverse_forward_source.composed_query_result_mismatch";
/// Detail: a composed response's family does not match its queries.
pub const DETAIL_COMPOSED_FAMILY: &str = "inverse_forward_source.composed_family_mismatch";
/// Detail: the numeric coding of a finite outcome is missing or inconsistent.
pub const DETAIL_COMPOSED_LEVELS: &str = "inverse_forward_source.composed_level_coding";
/// Detail: a composed mean was addressed by an outcome-law coordinate.
pub const DETAIL_COMPOSED_OUTCOME_COORDINATE: &str =
    "inverse_forward_source.composed_coordinate_is_outcome_law";
/// Detail: a Gaussian composition cannot supply the requested law.
pub const DETAIL_COMPOSED_LAW: &str = "inverse_forward_source.composed_law_not_materialized";
/// Detail: a point source was asked for a law.
pub const DETAIL_POINT_ONLY_LAW: &str = "inverse_forward_source.point_only_law_requested";
/// Detail: an interval region was requested of a source that has none.
pub const DETAIL_INTERVAL: &str = "inverse_forward_source.interval_not_licensed";
/// Detail: a per-action source cannot be sealed as one artifact.
pub const DETAIL_PER_ACTION: &str = "inverse_forward_source.per_action_not_sealable";
/// Detail: a stored origin does not describe the embedded claim.
pub const DETAIL_ORIGIN: &str = "inverse_forward_source.origin_invalid";

/// What kind of route produced the forward claim.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForwardSourceKind {
    /// A composition of fitted or declared mechanisms.
    ComposedMechanism,
    /// A cohort-mean treatment contrast in the model's own population.
    CohortContrast,
    /// A value carried to a target population.
    TransportedResponse,
}

impl ForwardSourceKind {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ComposedMechanism => "composed_mechanism",
            Self::CohortContrast => "cohort_contrast",
            Self::TransportedResponse => "transported_response",
        }
    }
}

/// What the claim supplies.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForwardRepresentation {
    /// One mean per coordinate.
    PointOnly,
    /// An exact or sampled law the engine can integrate.
    Law,
}

impl ForwardRepresentation {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PointOnly => "point_only",
            Self::Law => "law",
        }
    }
}

/// Where the forward claim came from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForwardOrigin {
    /// Version of this record.
    pub version: u16,
    /// The kind of producing route.
    pub kind: ForwardSourceKind,
    /// Identity of the producing object (composition premises, cohort source,
    /// transport receipt).
    pub source_identity: String,
    /// What the claim supplies.
    pub representation: ForwardRepresentation,
    /// Support label of the claim (`SupportStatus::as_str`).
    pub support: String,
    /// The transport receipt, present exactly for a transported response.
    pub transport: Option<TransportReceipt>,
    /// Mechanisms written down by the analyst rather than estimated; every
    /// claim of a composition rests on them as assumed.
    pub assumed_mechanisms: Vec<String>,
    /// Provider object that supplied the numbers.
    pub provider_id: String,
    /// Digest of the claim handed to the engine.
    pub claim_digest: String,
}

impl ForwardOrigin {
    /// Canonical identity: kind, source, representation, support, transport,
    /// assumed mechanisms (order-insensitive), provider and claim digest.
    #[must_use]
    pub fn canonical_identity(&self) -> String {
        let mut writer = CanonicalWriter::new(TAG_ORIGIN);
        writer
            .word(u64::from(self.version))
            .text(self.kind.as_str())
            .text(&self.source_identity)
            .text(self.representation.as_str())
            .text(&self.support)
            .opt_text(self.transport.as_ref().map(TransportReceipt::canonical_identity).as_deref());
        let mut assumed = self.assumed_mechanisms.clone();
        assumed.sort_unstable();
        assumed.dedup();
        writer.texts(&assumed).text(&self.provider_id).text(&self.claim_digest);
        writer.finish_hex()
    }

    /// Check the record is well formed.
    ///
    /// # Errors
    /// The reason it is not, as a static description.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.version != FORWARD_ORIGIN_VERSION {
            return Err("origin version is not this reader's");
        }
        if self.source_identity.trim().is_empty() || self.claim_digest.trim().is_empty() {
            return Err("origin names no source or claim");
        }
        if SupportStatus::from_name(&self.support).is_none() {
            return Err("origin support label is unknown");
        }
        match (self.kind, &self.transport) {
            (ForwardSourceKind::TransportedResponse, Some(receipt)) => {
                receipt.validate().map_err(|_| "origin transport identity is incomplete")?;
                if receipt.support != self.support
                    || receipt.source_identity.is_empty()
                    || receipt.canonical_identity() != self.source_identity
                {
                    return Err("origin does not match its transport receipt");
                }
            }
            (ForwardSourceKind::TransportedResponse, None) => {
                return Err("a transported origin carries its transport receipt");
            }
            (ForwardSourceKind::ComposedMechanism, Some(receipt)) => {
                // Mechanisms carried across populations: the receipt's label
                // governs, and its identity is bound beside the composition's.
                receipt.validate().map_err(|_| "origin transport identity is incomplete")?;
                if receipt.route != TransportedRoute::Mechanism || receipt.support != self.support {
                    return Err("origin does not match its transport receipt");
                }
            }
            (_, Some(_)) => {
                return Err("only a transported or carried origin carries a transport receipt");
            }
            (_, None) => {}
        }
        Ok(())
    }

    /// The same origin bound to another claim (the claim of one action).
    ///
    /// # Errors
    /// The claim cannot be encoded.
    pub fn bound_to(&self, claim: &ForwardClaim) -> Result<Self, IoError> {
        let mut bound = self.clone();
        bound.claim_digest = forward_claim_digest(claim)?;
        Ok(bound)
    }
}

/// Digest of a forward claim: the identities, the coordinates and every number.
///
/// # Errors
/// A law's metadata cannot be encoded.
pub fn forward_claim_digest(claim: &ForwardClaim) -> Result<String, IoError> {
    let mut writer = CanonicalWriter::new(TAG_CLAIM);
    match claim {
        ForwardClaim::Means(mean) => {
            writer
                .text("means")
                .text(&mean.provider_id)
                .text(&mean.snapshot_id)
                .text(&mean.causal_contract_id)
                .text(&mean.rng_id)
                .word(mean.coordinates.len() as u64);
            for coordinate in &mean.coordinates {
                coordinate.write_canonical(&mut writer);
            }
            writer.reals(&mean.means);
        }
        ForwardClaim::Law(law) => {
            let metadata = to_cbor(law.metadata())?;
            writer.text("law").bytes(&metadata).reals(law.draws());
        }
    }
    Ok(writer.finish_hex())
}

#[derive(Clone, Debug)]
enum SourceClaims {
    Single(ForwardClaim),
    PerAction(Vec<(String, ForwardClaim)>),
}

/// A forward source that passed its own checks, with the identity it carries.
#[derive(Clone, Debug)]
pub struct LicensedForwardSource {
    claims: SourceClaims,
    origin: ForwardOrigin,
}

impl LicensedForwardSource {
    /// Where the claim came from.
    #[must_use]
    pub const fn origin(&self) -> &ForwardOrigin {
        &self.origin
    }

    /// What the claim supplies.
    #[must_use]
    pub const fn representation(&self) -> ForwardRepresentation {
        self.origin.representation
    }

    /// The single claim shared by every action, when the source has one.
    #[must_use]
    pub fn claim(&self) -> Option<&ForwardClaim> {
        match &self.claims {
            SourceClaims::Single(claim) => Some(claim),
            SourceClaims::PerAction(_) => None,
        }
    }

    /// The claim of each action, when each action was answered separately.
    #[must_use]
    pub fn action_claims(&self) -> Option<&[(String, ForwardClaim)]> {
        match &self.claims {
            SourceClaims::Single(_) => None,
            SourceClaims::PerAction(claims) => Some(claims),
        }
    }
}

/// Why a forward source was refused.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ForwardSourceRefusal {
    /// The transported response itself was refused.
    Transported(TransportedRefusal),
    /// A transported response was offered where it is not accepted.
    TransportedNotAccepted,
    /// A composed response is malformed.
    ComposedInvalid(&'static str),
    /// A composed query and its stored result are different kinds.
    ComposedQueryResultMismatch(String),
    /// A composed response's family does not match its queries.
    ComposedFamilyMismatch(String),
    /// The numeric coding of a finite outcome is missing or inconsistent.
    ComposedLevelCoding {
        /// Action whose outcome levels are uncoded.
        action: String,
        /// What is wrong.
        problem: &'static str,
    },
    /// A composed mean is addressed by an outcome-law coordinate.
    ComposedCoordinateIsOutcomeLaw(String),
    /// A Gaussian composition cannot supply the requested law.
    ComposedLawNotMaterialized {
        /// What in the request needs the law.
        needed: &'static str,
    },
    /// A point source was asked for a law.
    PointOnlyLawRequested {
        /// The kind of source.
        kind: ForwardSourceKind,
        /// What in the request needs the law.
        needed: &'static str,
    },
    /// An interval region was requested of a source that has none.
    IntervalNotLicensed(ForwardSourceKind),
    /// A per-action source cannot be sealed as one artifact.
    PerActionNotSealable,
    /// The sealed artifact refused.
    Artifact {
        /// Registered code of the artifact refusal.
        code: &'static str,
        /// The refusal's text.
        text: String,
    },
    /// The inverse engine refused.
    Inverse(InverseQueryError),
}

impl ForwardSourceRefusal {
    /// Registered reason code.
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            Self::Transported(error) => error.code(),
            Self::TransportedNotAccepted | Self::IntervalNotLicensed(_) => {
                antecedent_core::reason_code!("cell_not_licensed")
            }
            Self::ComposedInvalid(_)
            | Self::ComposedFamilyMismatch(_)
            | Self::ComposedLevelCoding { .. } => {
                antecedent_core::reason_code!("mechanism_composition_incoherent")
            }
            Self::ComposedQueryResultMismatch(_)
            | Self::ComposedCoordinateIsOutcomeLaw(_)
            | Self::PerActionNotSealable => antecedent_core::reason_code!("invalid_argument"),
            Self::ComposedLawNotMaterialized { .. } | Self::PointOnlyLawRequested { .. } => {
                antecedent_core::reason_code!("decision_contract_unsatisfied")
            }
            Self::Artifact { code, .. } => *code,
            Self::Inverse(error) => error.reason_code(),
        }
    }

    /// Namespaced detail.
    #[must_use]
    pub fn detail(&self) -> String {
        match self {
            Self::Transported(error) => error.detail(),
            Self::TransportedNotAccepted => DETAIL_TRANSPORTED_NOT_ACCEPTED.to_owned(),
            Self::ComposedInvalid(_) => DETAIL_COMPOSED_INVALID.to_owned(),
            Self::ComposedQueryResultMismatch(_) => DETAIL_COMPOSED_MISMATCH.to_owned(),
            Self::ComposedFamilyMismatch(_) => DETAIL_COMPOSED_FAMILY.to_owned(),
            Self::ComposedLevelCoding { .. } => DETAIL_COMPOSED_LEVELS.to_owned(),
            Self::ComposedCoordinateIsOutcomeLaw(_) => {
                DETAIL_COMPOSED_OUTCOME_COORDINATE.to_owned()
            }
            Self::ComposedLawNotMaterialized { .. } => DETAIL_COMPOSED_LAW.to_owned(),
            Self::PointOnlyLawRequested { .. } => DETAIL_POINT_ONLY_LAW.to_owned(),
            Self::IntervalNotLicensed(_) => DETAIL_INTERVAL.to_owned(),
            Self::PerActionNotSealable => DETAIL_PER_ACTION.to_owned(),
            Self::Artifact { text, .. } => {
                text.split(": ").next().unwrap_or(DETAIL_ORIGIN).to_owned()
            }
            Self::Inverse(error) => error.to_refusal().detail,
        }
    }

    /// Structured refusal with a stated remedy.
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let build = |detail: &str| RefusalBuilder::new(self.code(), STAGE, detail);
        match self {
            Self::Transported(error) => crate::signal::at_stage(error.to_refusal(), STAGE),
            Self::TransportedNotAccepted => build(DETAIL_TRANSPORTED_NOT_ACCEPTED)
                .remedy(
                    "accept transported claims explicitly (AcceptsTransported) so the result \
                     carries the transport identity, or supply native target data",
                )
                .build(),
            Self::ComposedInvalid(what) => build(DETAIL_COMPOSED_INVALID).offending(*what).build(),
            Self::ComposedQueryResultMismatch(action) => build(DETAIL_COMPOSED_MISMATCH)
                .offending(action.clone())
                .remedy("export the composition query and its replayed result together")
                .build(),
            Self::ComposedFamilyMismatch(action) => build(DETAIL_COMPOSED_FAMILY)
                .offending(action.clone())
                .remedy("a response holds one family: linear-Gaussian or finite-table")
                .build(),
            Self::ComposedLevelCoding { action, problem } => build(DETAIL_COMPOSED_LEVELS)
                .offending(action.clone())
                .expected(*problem)
                .remedy(
                    "give every outcome level of the finite distribution exactly one numeric \
                     value; the composition has levels, not numbers",
                )
                .build(),
            Self::ComposedCoordinateIsOutcomeLaw(coordinate) => {
                build(DETAIL_COMPOSED_OUTCOME_COORDINATE)
                    .offending(coordinate.clone())
                    .remedy(
                        "address the interventional mean of the outcome (functional `mean`); \
                         an outcome-law coordinate needs a law, which a Gaussian composition \
                         does not hand over",
                    )
                    .build()
            }
            Self::ComposedLawNotMaterialized { needed } => build(DETAIL_COMPOSED_LAW)
                .offending(*needed)
                .remedy(
                    "draw the intervened joint law of the composition (the model's sampler) \
                     and supply the aligned draws as the forward law; the composition's \
                     closed form here is a mean and a variance",
                )
                .build(),
            Self::PointOnlyLawRequested { kind, needed } => build(DETAIL_POINT_ONLY_LAW)
                .offending(*needed)
                .supplied(kind.as_str())
                .remedy(
                    "a mean is one expectation; supply aligned outcome draws (an \
                     interventional predictive law) for a quantile, probability, nonlinear \
                     utility or outcome-law input",
                )
                .build(),
            Self::IntervalNotLicensed(kind) => build(DETAIL_INTERVAL)
                .supplied(kind.as_str())
                .remedy(
                    "none available: the source's own variance or standard error is an \
                     uncalibrated diagnostic and the transport step has no measured coverage; \
                     the value is point-only",
                )
                .build(),
            Self::PerActionNotSealable => build(DETAIL_PER_ACTION)
                .remedy("seal each action's query separately with seal_per_action_on_source")
                .build(),
            Self::Artifact { text, .. } => build(DETAIL_ORIGIN).offending(text.clone()).build(),
            Self::Inverse(error) => crate::signal::at_stage(error.to_refusal(), STAGE),
        }
    }
}

impl std::fmt::Display for ForwardSourceRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}: {}", antecedent_core::reason_code::PREFIX, self.code(), self.detail())
    }
}

impl std::error::Error for ForwardSourceRefusal {}

impl From<TransportedRefusal> for ForwardSourceRefusal {
    fn from(value: TransportedRefusal) -> Self {
        Self::Transported(value)
    }
}

impl From<InverseQueryError> for ForwardSourceRefusal {
    fn from(value: InverseQueryError) -> Self {
        Self::Inverse(value)
    }
}

fn artifact_refusal(error: &IoError) -> ForwardSourceRefusal {
    match error {
        IoError::Refused { code, message } => {
            ForwardSourceRefusal::Artifact { code: *code, text: message.clone() }
        }
        other => ForwardSourceRefusal::Artifact {
            code: antecedent_core::reason_code!("invalid_argument"),
            text: other.to_string(),
        },
    }
}

fn encoding(error: &IoError) -> ForwardSourceRefusal {
    artifact_refusal(error)
}

// ---------------------------------------------------------------------------
// Composed mechanisms
// ---------------------------------------------------------------------------

/// The family a composed response executed in.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComposedFamily {
    /// Additive-noise linear-Gaussian mechanisms: closed-form means and variances.
    LinearGaussian,
    /// Finite-table mechanisms: exact finite laws.
    Tabular,
}

impl ComposedFamily {
    const fn as_str(self) -> &'static str {
        match self {
            Self::LinearGaussian => "linear_gaussian",
            Self::Tabular => "tabular",
        }
    }
}

/// The replayed result of one composition query, for one action.
///
/// `query` and `result` are the pair of a consumed mechanism-composition
/// artifact; `premises_digest` is the digest that artifact verified.
#[derive(Clone, Debug, PartialEq)]
pub struct ComposedActionResponse {
    /// The declared action this response answers.
    pub action_id: String,
    /// The coordinate the action's utility reads (the interventional mean of
    /// the outcome, or the contrast).
    pub coordinate: ScientificQuantity,
    /// Verified premises digest of the composition artifact.
    pub premises_digest: String,
    /// The composition query.
    pub query: CompositionQueryWire,
    /// Its replayed result.
    pub result: CompositionResultWire,
    /// Numeric value of each outcome level, for a finite distribution only.
    pub level_values: Vec<(String, f64)>,
}

/// A composed-mechanism model's response over a grid of actions.
#[derive(Clone, Debug, PartialEq)]
pub struct ComposedResponse {
    /// The family every action's query executed in.
    pub family: ComposedFamily,
    /// Variables whose mechanism was declared rather than estimated.
    pub assumed_mechanisms: Vec<String>,
    /// Present when any component mechanism was carried across populations
    /// rather than refitted in the target: the receipt of that carrying (route
    /// `mechanism`, the populations, the selection-diagram digests and the
    /// declared assumptions, taken from the mechanism transport's own
    /// admissions). `None` states that every mechanism is native to the
    /// population the claim is about.
    pub transport: Option<TransportReceipt>,
    /// One response per answered action.
    pub actions: Vec<ComposedActionResponse>,
}

fn bits(value: u64) -> f64 {
    f64::from_bits(value)
}

fn combined_identity(response: &ComposedResponse) -> String {
    let mut rows: Vec<(&str, &str)> = response
        .actions
        .iter()
        .map(|a| (a.action_id.as_str(), a.premises_digest.as_str()))
        .collect();
    rows.sort_unstable();
    let mut writer = CanonicalWriter::new(TAG_COMPOSED);
    writer.text(response.family.as_str()).word(rows.len() as u64);
    for (action, digest) in rows {
        writer.text(action).text(digest);
    }
    writer.finish_hex()
}

fn point_value(action: &ComposedActionResponse) -> Result<f64, ForwardSourceRefusal> {
    let mismatch = || ForwardSourceRefusal::ComposedQueryResultMismatch(action.action_id.clone());
    let value = match (&action.query, &action.result) {
        (
            CompositionQueryWire::GaussianInterventionLaw { .. },
            CompositionResultWire::GaussianLaw { mean_bits, variance_bits },
        ) => {
            if !bits(*variance_bits).is_finite() || bits(*variance_bits) < 0.0 {
                return Err(ForwardSourceRefusal::ComposedInvalid(
                    "variance is finite and non-negative",
                ));
            }
            bits(*mean_bits)
        }
        (
            CompositionQueryWire::GaussianMeanContrast { .. },
            CompositionResultWire::GaussianFunctional { point_bits, variance_bits, .. },
        ) => {
            if !bits(*variance_bits).is_finite() || bits(*variance_bits) < 0.0 {
                return Err(ForwardSourceRefusal::ComposedInvalid(
                    "variance is finite and non-negative",
                ));
            }
            bits(*point_bits)
        }
        _ => return Err(mismatch()),
    };
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ForwardSourceRefusal::ComposedInvalid("composed value is finite"))
    }
}

fn finite_law(
    action: &ComposedActionResponse,
    provider_id: &str,
    causal_contract_id: &str,
) -> Result<DistributionArtifact, ForwardSourceRefusal> {
    let mismatch = || ForwardSourceRefusal::ComposedQueryResultMismatch(action.action_id.clone());
    let (
        CompositionQueryWire::FiniteInterventionDistribution { .. },
        CompositionResultWire::FiniteDistribution { probabilities },
    ) = (&action.query, &action.result)
    else {
        return Err(mismatch());
    };
    let coding = |problem| ForwardSourceRefusal::ComposedLevelCoding {
        action: action.action_id.clone(),
        problem,
    };
    if probabilities.is_empty() {
        return Err(coding("the distribution has at least one level"));
    }
    let mut values = Vec::with_capacity(probabilities.len());
    let mut weights = Vec::with_capacity(probabilities.len());
    for (level, probability_bits) in probabilities {
        let probability = bits(*probability_bits);
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(ForwardSourceRefusal::ComposedInvalid("level probability is in [0, 1]"));
        }
        let mut matches = action.level_values.iter().filter(|(name, _)| name == level);
        let Some((_, value)) = matches.next() else {
            return Err(coding("every outcome level has a numeric value"));
        };
        if matches.next().is_some() {
            return Err(coding("each outcome level has one numeric value"));
        }
        if !value.is_finite() {
            return Err(coding("numeric values are finite"));
        }
        values.push(*value);
        weights.push(probability);
    }
    if action.level_values.len() != probabilities.len() {
        return Err(coding("the coding names only levels of the distribution"));
    }
    let total: f64 = weights.iter().sum();
    if (total - 1.0).abs() > PROBABILITY_SUM_TOLERANCE {
        return Err(ForwardSourceRefusal::ComposedInvalid("level probabilities sum to one"));
    }
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        std::slice::from_ref(&action.coordinate),
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: action.premises_digest.clone(),
            provider_id: provider_id.to_owned(),
            rng_id: "deterministic_exact".to_owned(),
            snapshot_id: action.premises_digest.clone(),
            causal_contract_id: causal_contract_id.to_owned(),
        },
    )
    .map_err(|error| encoding(&error))?;
    let rows = values.len();
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [rows, 1],
            weights: Some(weights),
            supported: None,
            // Exact given the declared mechanisms, but those mechanisms were
            // estimated or assumed: no calibration is claimed and no standard
            // error of zero is implied.
            calibration: DistributionCalibration::PointOnly,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        values,
    )
    .map_err(|error| encoding(&error))
}

/// Adapt a composed-mechanism model's response as a forward source.
///
/// A linear-Gaussian response becomes one mean source over every action's
/// coordinate; a finite-table response becomes one exact finite law per action
/// (each action is its own intervention, so no pairing across actions exists).
///
/// A response whose mechanisms were carried across populations is a
/// transported claim and is refused here, as every transported source is
/// refused where only native target data is accepted; read it through
/// [`composed_forward_source_accepting`] with `AcceptsTransported`.
///
/// # Errors
/// No action, repeated action or coordinate, a blank premises digest, a query
/// and result of different kinds, a family that does not match the queries, a
/// mean addressed by an outcome-law coordinate, a non-finite value, a finite
/// outcome whose levels are not coded exactly once, and a response with carried
/// mechanisms ([`ForwardSourceRefusal::TransportedNotAccepted`]).
pub fn composed_forward_source(
    response: &ComposedResponse,
) -> Result<LicensedForwardSource, ForwardSourceRefusal> {
    composed_forward_source_accepting(response, TransportedInputPolicy::NativeTargetOnly)
}

/// [`composed_forward_source`] with the consumer's stance on transported
/// claims stated.
///
/// A response with carried mechanisms is read only under
/// `AcceptsTransported`; its origin then carries the transport receipt and the
/// receipt's support label (never `supported`), exactly as a transported
/// response does. A response with no carried mechanisms is a native
/// composition under either policy.
///
/// # Errors
/// Those of [`composed_forward_source`]; a receipt that is malformed, whose
/// route is not `mechanism`, or whose support is `supported`.
pub fn composed_forward_source_accepting(
    response: &ComposedResponse,
    accept: TransportedInputPolicy,
) -> Result<LicensedForwardSource, ForwardSourceRefusal> {
    if let Some(receipt) = &response.transport {
        if accept == TransportedInputPolicy::NativeTargetOnly {
            return Err(ForwardSourceRefusal::TransportedNotAccepted);
        }
        receipt.validate()?;
        if receipt.route != TransportedRoute::Mechanism {
            return Err(ForwardSourceRefusal::ComposedInvalid(
                "a carried composition's receipt names the mechanism route",
            ));
        }
    }
    if response.actions.is_empty() {
        return Err(ForwardSourceRefusal::ComposedInvalid("at least one action"));
    }
    for (position, action) in response.actions.iter().enumerate() {
        if action.action_id.trim().is_empty() || action.premises_digest.trim().is_empty() {
            return Err(ForwardSourceRefusal::ComposedInvalid("action and premises identities"));
        }
        if response.actions[..position].iter().any(|earlier| {
            earlier.action_id == action.action_id
                || earlier.coordinate.require_same_coordinate(&action.coordinate).is_ok()
        }) {
            return Err(ForwardSourceRefusal::ComposedInvalid(
                "actions and coordinates are distinct",
            ));
        }
        let gaussian = matches!(
            action.query,
            CompositionQueryWire::GaussianInterventionLaw { .. }
                | CompositionQueryWire::GaussianMeanContrast { .. }
        );
        if gaussian != (response.family == ComposedFamily::LinearGaussian) {
            return Err(ForwardSourceRefusal::ComposedFamilyMismatch(action.action_id.clone()));
        }
    }
    let identity = combined_identity(response);
    let provider_id = format!("composed_mechanisms:{}", response.family.as_str());
    let causal_contract_id = format!("composition:{identity}");
    let (claims, representation) = match response.family {
        ComposedFamily::LinearGaussian => {
            let mut coordinates = Vec::with_capacity(response.actions.len());
            let mut means = Vec::with_capacity(response.actions.len());
            for action in &response.actions {
                if action.coordinate.functional_id == "outcome" {
                    return Err(ForwardSourceRefusal::ComposedCoordinateIsOutcomeLaw(
                        action.coordinate.variable_id.clone(),
                    ));
                }
                means.push(point_value(action)?);
                coordinates.push(action.coordinate.clone());
            }
            (
                SourceClaims::Single(ForwardClaim::Means(MeanSource {
                    coordinates,
                    means,
                    provider_id: provider_id.clone(),
                    snapshot_id: identity.clone(),
                    causal_contract_id,
                    rng_id: "none:composed_closed_form".to_owned(),
                })),
                ForwardRepresentation::PointOnly,
            )
        }
        ComposedFamily::Tabular => {
            let mut claims = Vec::with_capacity(response.actions.len());
            for action in &response.actions {
                let law = finite_law(action, &provider_id, &causal_contract_id)?;
                claims.push((action.action_id.clone(), ForwardClaim::Law(Box::new(law))));
            }
            (SourceClaims::PerAction(claims), ForwardRepresentation::Law)
        }
    };
    let claim_digest = match &claims {
        SourceClaims::Single(claim) => forward_claim_digest(claim).map_err(|e| encoding(&e))?,
        SourceClaims::PerAction(per_action) => {
            let mut writer = CanonicalWriter::new(TAG_CLAIM);
            writer.text("per_action").word(per_action.len() as u64);
            let mut rows: Vec<&(String, ForwardClaim)> = per_action.iter().collect();
            rows.sort_by(|a, b| a.0.cmp(&b.0));
            for (action, claim) in rows {
                writer.text(action).text(&forward_claim_digest(claim).map_err(|e| encoding(&e))?);
            }
            writer.finish_hex()
        }
    };
    // A composition with carried mechanisms is as strong as its transport:
    // the receipt's label governs, never `supported`.
    let support = response
        .transport
        .as_ref()
        .map_or_else(|| SupportStatus::Supported.as_str().to_owned(), |t| t.support.clone());
    Ok(LicensedForwardSource {
        claims,
        origin: ForwardOrigin {
            version: FORWARD_ORIGIN_VERSION,
            kind: ForwardSourceKind::ComposedMechanism,
            source_identity: identity,
            representation,
            support,
            transport: response.transport.clone(),
            assumed_mechanisms: response.assumed_mechanisms.clone(),
            provider_id,
            claim_digest,
        },
    })
}

// ---------------------------------------------------------------------------
// Cohort contrast and transported responses
// ---------------------------------------------------------------------------

/// Adapt a cohort-mean contrast as a forward source.
///
/// A source in the model's own population is a native cohort contrast. A
/// source built under a declared covariate shift is a transported response: it
/// is read only under [`TransportedInputPolicy::AcceptsTransported`], and the
/// result then carries its transport identity.
///
/// # Errors
/// [`ForwardSourceRefusal::TransportedNotAccepted`] for a shifted cohort under
/// `NativeTargetOnly`, and every refusal of [`transported_forward_source`].
pub fn cohort_forward_source(
    source: &CohortDecisionSource,
    accept: TransportedInputPolicy,
) -> Result<LicensedForwardSource, ForwardSourceRefusal> {
    if source.shift().is_some() {
        if accept == TransportedInputPolicy::NativeTargetOnly {
            return Err(ForwardSourceRefusal::TransportedNotAccepted);
        }
        let response = TransportedResponse::from_shifted_cohort(source)?;
        return transported_forward_source(&response, accept);
    }
    let claim = ForwardClaim::Means(cohort_mean_source(source));
    let claim_digest = forward_claim_digest(&claim).map_err(|e| encoding(&e))?;
    Ok(LicensedForwardSource {
        origin: ForwardOrigin {
            version: FORWARD_ORIGIN_VERSION,
            kind: ForwardSourceKind::CohortContrast,
            source_identity: source.identity().to_owned(),
            representation: ForwardRepresentation::PointOnly,
            support: source.support_status().as_str().to_owned(),
            transport: None,
            assumed_mechanisms: Vec::new(),
            provider_id: source.provider_id(),
            claim_digest,
        },
        claims: SourceClaims::Single(claim),
    })
}

/// Adapt a transported response as a forward source.
///
/// # Errors
/// [`ForwardSourceRefusal::TransportedNotAccepted`] under `NativeTargetOnly`.
pub fn transported_forward_source(
    response: &TransportedResponse,
    accept: TransportedInputPolicy,
) -> Result<LicensedForwardSource, ForwardSourceRefusal> {
    if accept == TransportedInputPolicy::NativeTargetOnly {
        return Err(ForwardSourceRefusal::TransportedNotAccepted);
    }
    let (claim, provider_id) = match response.value() {
        TransportedValue::Means(mean) => {
            (ForwardClaim::Means(mean.clone()), mean.provider_id.clone())
        }
        TransportedValue::Law(law) => {
            (ForwardClaim::Law(Box::new(law.clone())), law.metadata().identity.provider_id.clone())
        }
    };
    let receipt = response.receipt();
    let claim_digest = forward_claim_digest(&claim).map_err(|e| encoding(&e))?;
    Ok(LicensedForwardSource {
        origin: ForwardOrigin {
            version: FORWARD_ORIGIN_VERSION,
            kind: ForwardSourceKind::TransportedResponse,
            source_identity: receipt.canonical_identity(),
            representation: match receipt.representation {
                TransportedRepresentation::PointOnly => ForwardRepresentation::PointOnly,
                TransportedRepresentation::Law => ForwardRepresentation::Law,
            },
            support: receipt.support.clone(),
            transport: Some(receipt.clone()),
            assumed_mechanisms: Vec::new(),
            provider_id,
            claim_digest,
        },
        claims: SourceClaims::Single(claim),
    })
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

/// What an inverse request asks of its forward source beyond the query itself.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InverseDemand {
    /// Whether the caller wants feasibility over a published interval region.
    pub interval_region: bool,
}

/// An inverse result with the identity of the source it stands on.
#[derive(Clone, Debug, PartialEq)]
pub struct SourcedInverseResult {
    /// The engine's result.
    pub result: InverseResult,
    /// Where the forward claim came from.
    pub origin: ForwardOrigin,
}

/// The first thing in `query` that a mean cannot answer, when there is one.
#[must_use]
pub fn query_needs_law(query: &InverseQuery) -> Option<&'static str> {
    for constraint in &query.constraints {
        match constraint {
            InverseConstraint::TargetQuantile { .. } => return Some("quantile"),
            InverseConstraint::ProbabilityThreshold { .. } => return Some("probability"),
            InverseConstraint::TargetMean { .. } => {}
        }
    }
    action_needs_law(&query.contract, &query.grid_order)
}

fn action_needs_law(contract: &DecisionContract, grid: &[String]) -> Option<&'static str> {
    for id in grid {
        let Some(action) = contract.actions.iter().find(|a| &a.id == id) else {
            continue;
        };
        if !action.utility.is_affine() {
            return Some("nonlinear_utility");
        }
        if action.inputs.iter().any(|q| q.functional_id == "outcome") {
            return Some("outcome_law_input");
        }
    }
    None
}

fn check_demand(
    query: &InverseQuery,
    source: &LicensedForwardSource,
    demand: InverseDemand,
) -> Result<(), ForwardSourceRefusal> {
    let kind = source.origin.kind;
    if demand.interval_region {
        return Err(ForwardSourceRefusal::IntervalNotLicensed(kind));
    }
    if source.origin.representation == ForwardRepresentation::PointOnly {
        if let Some(needed) = query_needs_law(query) {
            return Err(match kind {
                ForwardSourceKind::ComposedMechanism => {
                    ForwardSourceRefusal::ComposedLawNotMaterialized { needed }
                }
                _ => ForwardSourceRefusal::PointOnlyLawRequested { kind, needed },
            });
        }
    }
    Ok(())
}

/// Evaluate an inverse query on a licensed forward source.
///
/// The request is checked against what the source supplies before the engine
/// runs: an interval region is refused for every source, and a quantile,
/// probability, nonlinear utility or outcome-law input is refused for a point.
/// The engine then compares the claim's value with the target exactly as it
/// does for any claim; the result is returned beside the source's origin.
///
/// # Errors
/// [`ForwardSourceRefusal::IntervalNotLicensed`],
/// [`ForwardSourceRefusal::PointOnlyLawRequested`],
/// [`ForwardSourceRefusal::ComposedLawNotMaterialized`], and any refusal of the
/// inverse engine.
pub fn evaluate_inverse_on_source(
    query: &InverseQuery,
    source: &LicensedForwardSource,
    demand: InverseDemand,
) -> Result<SourcedInverseResult, ForwardSourceRefusal> {
    check_demand(query, source, demand)?;
    let result = match &source.claims {
        SourceClaims::Single(claim) => {
            let evidence =
                ForwardEvidence { point: Some(claim.clone()), ..ForwardEvidence::default() };
            evaluate_inverse_query(query, &evidence)?
        }
        SourceClaims::PerAction(claims) => evaluate_inverse_query_per_action(query, claims)?,
    };
    Ok(SourcedInverseResult { result, origin: source.origin.clone() })
}

/// Evaluate and seal a query on a source that has one claim for every action,
/// binding the source's origin into the artifact.
///
/// # Errors
/// [`ForwardSourceRefusal::PerActionNotSealable`] for a source that answers
/// each action separately, the refusals of [`evaluate_inverse_on_source`], and
/// the artifact's own refusals.
pub fn seal_on_source(
    query: InverseQuery,
    source: &LicensedForwardSource,
    demand: InverseDemand,
) -> Result<InverseQueryArtifact, ForwardSourceRefusal> {
    check_demand(&query, source, demand)?;
    let SourceClaims::Single(claim) = &source.claims else {
        return Err(ForwardSourceRefusal::PerActionNotSealable);
    };
    let evidence = ForwardEvidence { point: Some(claim.clone()), ..ForwardEvidence::default() };
    InverseQueryArtifact::new(query, evidence)?
        .with_forward_origin(source.origin.clone())
        .map_err(|error| artifact_refusal(&error))
}

/// Evaluate and seal one single-action artifact per answered action, each
/// bound to its own claim and the source's origin.
///
/// # Errors
/// As [`seal_on_source`], for a source that answers each action separately.
pub fn seal_per_action_on_source(
    query: &InverseQuery,
    source: &LicensedForwardSource,
    demand: InverseDemand,
) -> Result<Vec<(String, InverseQueryArtifact)>, ForwardSourceRefusal> {
    check_demand(query, source, demand)?;
    let Some(claims) = source.action_claims() else {
        return Err(ForwardSourceRefusal::PerActionNotSealable);
    };
    let mut sealed = Vec::with_capacity(claims.len());
    for (action, claim) in claims {
        if !query.grid_order.contains(action) {
            continue;
        }
        let mut single = query.clone();
        single.grid_order = vec![action.clone()];
        let evidence = ForwardEvidence { point: Some(claim.clone()), ..ForwardEvidence::default() };
        let origin = source.origin.bound_to(claim).map_err(|e| encoding(&e))?;
        let artifact = InverseQueryArtifact::new(single, evidence)?
            .with_forward_origin(origin)
            .map_err(|error| artifact_refusal(&error))?;
        sealed.push((action.clone(), artifact));
    }
    Ok(sealed)
}
