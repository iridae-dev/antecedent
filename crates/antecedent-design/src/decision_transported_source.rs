//! A transported response as a licensed decision source.
//!
//! A response carried to a target population (a cohort contrast under a declared
//! covariate shift, a policy value under a selection diagram, a synthesis that
//! contains transported claims, a mechanism-level transport) reaches the
//! decision boundary as a [`TransportedResponse`]: its value, the
//! [`TransportReceipt`] that says how it was carried, and nothing else. The
//! receipt is the transport identity (source populations, target population,
//! selection-diagram identities, declared assumptions); it travels with the
//! value into the decision result, so a decision on a transported response is
//! never presented as a decision on native target data.
//!
//! # What a transported response can and cannot supply
//!
//! - **Point only.** A transported mean supplies the expectation of an affine
//!   utility and nothing else. A request that needs a law (a probability, a
//!   quantile, a nonlinear utility, a hard constraint, regret) is refused with
//!   [`TransportedRefusal::LawNotSupplied`]; the value is never widened into a
//!   law by assuming a spread.
//! - **Law.** A transported law (aligned draws under a declared transport
//!   model) answers every functional the shared engine answers. Its support
//!   label is the receipt's, never the artifact's own mask alone.
//! - **Interval.** Neither representation carries an interval: the transport
//!   step and the support check have no measured coverage. An interval request
//!   is refused with [`TransportedRefusal::IntervalNotLicensed`].
//!
//! # Acceptance
//!
//! A transported response is labelled at best `Extrapolative`
//! ([`TransportedResponse::new`] refuses a stronger label or a native or
//! calibrated label on the value). The default support policy admits only
//! `Supported` coordinates, so a decision on a transported response needs a
//! policy that names the weaker status explicitly; that is the consumer's
//! acceptance of a transported claim, and the refusal when it is absent is
//! [`TransportedRefusal::SupportBelowPolicy`].
//!
//! The standard error of a source model, when one exists, is retained on the
//! receipt as `uncalibrated_standard_error`. It is a diagnostic: no decision,
//! interval or feasibility statement inherits it.
//!
//! # Ownership
//!
//! A response owns its value and receipt. Evaluation borrows the contract and
//! the response and retains neither.
//!
//! # Refresh
//!
//! Refresh with outcome-only data. A transported response is a value and the
//! receipt of how it was carried. New outcome-only data at the source
//! population changes the carried value and so its source identity: the
//! response is rebuilt from the producing result and the decision is
//! recomputed. The declaration (populations, selection-diagram identities,
//! assumptions) and the consumer's support policy are kept. Data of another
//! population, quantity or arm is refused (`coordinate_not_in_target`,
//! `mixed_transports`); the receipt is never edited in place to follow new
//! data.
//!
//! # Export
//!
//! A transported point and its decision are exported and consumed by
//! `transported_decision_artifact`, which re-derives the decision from the
//! stored contract, value, receipt and policy rather than trusting the stored
//! numbers. Routes with a typed constructor here (shifted cohort, policy,
//! synthesis) cannot be declared by hand: their populations and identities are
//! copied from the producing result. A typed route's transport identities must
//! be 32-byte digests; only the `declared` route carries free text.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_contract::{ExternalRefusal, ScientificQuantity};
use antecedent_core::{CanonicalWriter, SupportStatus};
use antecedent_io_core::cohort_decision::CohortDecisionSource;
use antecedent_io_core::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionTrust,
};
use serde::{Deserialize, Serialize};

use crate::cohort_decision_input::cohort_mean_source;
use crate::composition_boundary::{
    BoundaryError, DecisionInput, SupportPolicy, SupportedDecision, TrustEvidence,
    TrustRequirement, evaluate_with_support,
};
use crate::decision_contract::{DecisionContract, DecisionCriterion};
use crate::decision_eval::MeanSource;
use crate::signal::RefusalBuilder;
use crate::transported_policy::TransportedPolicyEvaluation;
use crate::transported_synthesis::TransportedSynthesis;

/// Version of the receipt and response identity encodings. Bump only when a
/// canonical encoding in this module changes.
pub const TRANSPORTED_RESPONSE_VERSION: u16 = 1;
/// The only inferential claim a transported response carries.
pub const TRANSPORTED_RESPONSE_INFERENCE: &str = "point_only";

const TAG_RECEIPT: &str = "antecedent.transported_response.receipt";
const TAG_DECISION: &str = "antecedent.transported_response.decision";
const TAG_MERGE: &str = "antecedent.transported_response.merge";
const TAG_SHIFT: &str = "antecedent.transported_response.covariate_shift";
const STAGE: &str = "transported_decision";

/// Detail: the transport identity names no source, target or assumption.
pub const DETAIL_IDENTITY_INCOMPLETE: &str = "transported_decision.transport_identity_incomplete";
/// Detail: a cohort or policy that was not carried anywhere was offered as transported.
pub const DETAIL_NOT_TRANSPORTED: &str = "transported_decision.not_transported";
/// Detail: a coordinate is not stated in the target population.
pub const DETAIL_NOT_IN_TARGET: &str = "transported_decision.coordinate_not_in_target";
/// Detail: a transported value was labelled `Supported`.
pub const DETAIL_SUPPORT_INFLATED: &str = "transported_decision.support_label_inflated";
/// Detail: a transported law was labelled exact or measured.
pub const DETAIL_CALIBRATION_INFLATED: &str = "transported_decision.calibration_label_inflated";
/// Detail: a transported law was labelled native.
pub const DETAIL_NATIVE_LABEL: &str = "transported_decision.native_label_on_transported_value";
/// Detail: the value is malformed.
pub const DETAIL_INVALID_VALUE: &str = "transported_decision.invalid_value";
/// Detail: the request needs a law and the transported value is a point.
pub const DETAIL_LAW_NOT_SUPPLIED: &str = "transported_decision.law_not_supplied";
/// Detail: the request needs an interval.
pub const DETAIL_INTERVAL_NOT_LICENSED: &str = "transported_decision.interval_not_licensed";
/// Detail: responses merged into one grid were not carried by one transport.
pub const DETAIL_MIXED_TRANSPORTS: &str = "transported_decision.mixed_transports";
/// Detail: a transport identity of a typed route is not a 32-byte digest.
pub const DETAIL_IDENTITY_NOT_DIGEST: &str = "transported_decision.transport_identity_not_digest";
/// Detail: a route that has a typed constructor was declared by hand.
pub const DETAIL_ROUTE_NEEDS_TYPED_SOURCE: &str =
    "transported_decision.route_needs_typed_source";
/// Detail: the consumer's policy does not accept the transported support
/// label (the decision-side form of `inverse_forward_source.transported_not_accepted`).
pub const DETAIL_SUPPORT_BELOW_POLICY: &str = "transported_decision.transported_not_accepted";

/// The route a transported value was carried by.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransportedRoute {
    /// A retained effect model's cohort mean under a declared selection shift.
    CohortEffect,
    /// A cohort-mean contrast under a declared covariate shift.
    ShiftedCohort,
    /// A fixed policy evaluated in a target population.
    Policy,
    /// A synthesis that contains transported claims.
    Synthesis,
    /// Mechanism-level transport of fitted mechanisms.
    Mechanism,
    /// Another route, declared by the caller with its own identity.
    Declared,
}

impl TransportedRoute {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CohortEffect => "cohort_effect",
            Self::ShiftedCohort => "shifted_cohort",
            Self::Policy => "policy",
            Self::Synthesis => "synthesis",
            Self::Mechanism => "mechanism",
            Self::Declared => "declared",
        }
    }
}

/// What a transported value is.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransportedRepresentation {
    /// One mean per coordinate and nothing else.
    PointOnly,
    /// Aligned draws of a law under a declared transport model.
    Law,
}

impl TransportedRepresentation {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PointOnly => "point_only",
            Self::Law => "law",
        }
    }
}

/// How a value was carried to its target population.
///
/// Every field is a declaration or an identity of the producing route; none is
/// evidence that the premises hold.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportReceipt {
    /// The route that carried the value.
    pub route: TransportedRoute,
    /// Identity of the transported object (the producing route's own identity).
    pub source_identity: String,
    /// Populations the claim was learned in.
    pub source_populations: Vec<String>,
    /// Population the claim was carried to.
    pub target_population: String,
    /// Identities of the selection diagrams or shift declarations, sorted and
    /// distinct. A synthesis of several transports lists each.
    pub transport_identities: Vec<String>,
    /// Declared invariance and selection assumptions, sorted and distinct.
    pub assumptions: Vec<String>,
    /// Support label (`SupportStatus::as_str`); never `supported`.
    pub support: String,
    /// What the value is.
    pub representation: TransportedRepresentation,
    /// The source model's own standard error, when it has one: a diagnostic,
    /// never an interval and never inherited by a decision.
    pub uncalibrated_standard_error: Option<f64>,
}

impl TransportReceipt {
    /// The receipt's support label as a status.
    #[must_use]
    pub fn support_status(&self) -> Option<SupportStatus> {
        SupportStatus::from_name(&self.support)
    }

    /// Check the receipt is a well formed transport identity.
    ///
    /// # Errors
    /// [`TransportedRefusal::TransportIdentityIncomplete`] for a blank identity,
    /// no source population (or one equal to the target), no assumption or no
    /// transport identity; [`TransportedRefusal::SupportLabelInflated`] for a
    /// label that is unknown or stronger than `extrapolative`.
    pub fn validate(&self) -> Result<(), TransportedRefusal> {
        let blank = |text: &str| text.trim().is_empty();
        if blank(&self.source_identity)
            || blank(&self.target_population)
            || self.source_populations.is_empty()
            || self.source_populations.iter().any(|p| blank(p))
            || self.source_populations.contains(&self.target_population)
            || self.assumptions.is_empty()
            || self.assumptions.iter().any(|a| blank(a))
            || self.transport_identities.is_empty()
            || self.transport_identities.iter().any(|t| blank(t))
        {
            return Err(TransportedRefusal::TransportIdentityIncomplete);
        }
        // A typed route's identities come from a producing route's canonical
        // digests; only a hand-declared route may carry free text.
        if self.route != TransportedRoute::Declared
            && !self.transport_identities.iter().all(|t| is_digest_hex(t))
        {
            return Err(TransportedRefusal::IdentityNotDigest);
        }
        match self.support_status() {
            Some(status) if status.severity() >= SupportStatus::Extrapolative.severity() => {}
            _ => return Err(TransportedRefusal::SupportLabelInflated(self.support.clone())),
        }
        if self.uncalibrated_standard_error.is_some_and(|e| !e.is_finite() || e < 0.0) {
            return Err(TransportedRefusal::InvalidValue("uncalibrated_standard_error"));
        }
        Ok(())
    }

    /// Canonical identity of the receipt: populations, identities and
    /// assumptions are order-insensitive.
    #[must_use]
    pub fn canonical_identity(&self) -> String {
        let mut writer = CanonicalWriter::new(TAG_RECEIPT);
        writer
            .word(u64::from(TRANSPORTED_RESPONSE_VERSION))
            .text(self.route.as_str())
            .text(&self.source_identity);
        let sorted = |values: &[String]| {
            let mut out = values.to_vec();
            out.sort_unstable();
            out.dedup();
            out
        };
        writer
            .texts(&sorted(&self.source_populations))
            .text(&self.target_population)
            .texts(&sorted(&self.transport_identities))
            .texts(&sorted(&self.assumptions))
            .text(&self.support)
            .text(self.representation.as_str())
            .opt_real(self.uncalibrated_standard_error);
        writer.finish_hex()
    }
}

fn is_digest_hex(text: &str) -> bool {
    text.len() == 64 && text.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// The value of a transported response.
#[derive(Clone, Debug, PartialEq)]
// The law is the large variant; boxing it would change the public pattern.
#[allow(clippy::large_enum_variant, reason = "built once per call and moved, never stored in bulk")]
pub enum TransportedValue {
    /// One mean per coordinate.
    Means(MeanSource),
    /// Aligned draws under a declared transport model.
    Law(DistributionArtifact),
}

/// What a route declares when it hands a value to the decision boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportDeclaration {
    /// The route that carried the value.
    pub route: TransportedRoute,
    /// Identity of the transported object.
    pub source_identity: String,
    /// Populations the claim was learned in.
    pub source_populations: Vec<String>,
    /// Population the claim was carried to.
    pub target_population: String,
    /// Selection-diagram or shift-declaration identities.
    pub transport_identities: Vec<String>,
    /// Declared invariance and selection assumptions.
    pub assumptions: Vec<String>,
    /// Support label; defaults to `Extrapolative` through the constructors.
    pub support: SupportStatus,
    /// The source model's own standard error, when it has one.
    pub uncalibrated_standard_error: Option<f64>,
    /// Provider object that supplied the value.
    pub provider_id: String,
    /// Snapshot or exact-request identity the value was computed from.
    pub snapshot_id: String,
    /// Causal-contract identity the value was bound to.
    pub causal_contract_id: String,
}

/// A value carried to a target population, with the identity of the transport.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportedResponse {
    value: TransportedValue,
    receipt: TransportReceipt,
}

fn finalize(
    declaration: &TransportDeclaration,
    representation: TransportedRepresentation,
) -> TransportReceipt {
    let sorted = |values: &[String]| -> Vec<String> {
        values.iter().cloned().collect::<BTreeSet<_>>().into_iter().collect()
    };
    TransportReceipt {
        route: declaration.route,
        source_identity: declaration.source_identity.clone(),
        source_populations: sorted(&declaration.source_populations),
        target_population: declaration.target_population.clone(),
        transport_identities: sorted(&declaration.transport_identities),
        assumptions: sorted(&declaration.assumptions),
        support: declaration.support.as_str().to_owned(),
        representation,
        uncalibrated_standard_error: declaration.uncalibrated_standard_error,
    }
}

impl TransportedResponse {
    /// A transported point: one mean per coordinate.
    ///
    /// # Errors
    /// An incomplete transport identity; a support label stronger than
    /// `extrapolative`; no coordinate, a repeated coordinate, a count mismatch
    /// or a non-finite mean; a coordinate not stated in the target population.
    pub fn declared_point(
        coordinates: Vec<ScientificQuantity>,
        means: Vec<f64>,
        declaration: &TransportDeclaration,
    ) -> Result<Self, TransportedRefusal> {
        require_declarable_route(declaration.route)?;
        Self::build_point(coordinates, means, declaration)
    }

    fn build_point(
        coordinates: Vec<ScientificQuantity>,
        means: Vec<f64>,
        declaration: &TransportDeclaration,
    ) -> Result<Self, TransportedRefusal> {
        if coordinates.is_empty()
            || coordinates.len() != means.len()
            || means.iter().any(|m| !m.is_finite())
        {
            return Err(TransportedRefusal::InvalidValue("one finite mean per coordinate"));
        }
        for (position, coordinate) in coordinates.iter().enumerate() {
            if coordinates[..position]
                .iter()
                .any(|earlier| earlier.require_same_coordinate(coordinate).is_ok())
            {
                return Err(TransportedRefusal::InvalidValue("coordinates are distinct"));
            }
        }
        let receipt = finalize(declaration, TransportedRepresentation::PointOnly);
        receipt.validate()?;
        check_in_target(&coordinates, &receipt.target_population)?;
        Ok(Self {
            value: TransportedValue::Means(MeanSource {
                coordinates,
                means,
                provider_id: declaration.provider_id.clone(),
                snapshot_id: declaration.snapshot_id.clone(),
                causal_contract_id: declaration.causal_contract_id.clone(),
                rng_id: "none:transported_point".to_owned(),
            }),
            receipt,
        })
    }

    /// A transported law: aligned draws under a declared transport model.
    ///
    /// # Errors
    /// As [`Self::declared_point`], and a law labelled native, exact or
    /// measured (a transport never raises a calibration or trust label).
    pub fn declared_law(
        law: DistributionArtifact,
        declaration: &TransportDeclaration,
    ) -> Result<Self, TransportedRefusal> {
        require_declarable_route(declaration.route)?;
        let meta = law.metadata();
        if meta.trust == DistributionTrust::NativeLicensed {
            return Err(TransportedRefusal::NativeLabelOnTransported);
        }
        if matches!(
            meta.calibration,
            DistributionCalibration::Exact | DistributionCalibration::Measured
        ) {
            return Err(TransportedRefusal::CalibrationLabelInflated);
        }
        let coordinates: Vec<ScientificQuantity> = law
            .quantities()
            .iter()
            .map(|wire| ScientificQuantity::try_from(wire.clone()))
            .collect::<Result<_, _>>()
            .map_err(|_| TransportedRefusal::InvalidValue("law coordinate"))?;
        let receipt = finalize(declaration, TransportedRepresentation::Law);
        receipt.validate()?;
        check_in_target(&coordinates, &receipt.target_population)?;
        Ok(Self { value: TransportedValue::Law(law), receipt })
    }

    /// The cohort-mean contrast of a source built under a declared covariate
    /// shift.
    ///
    /// # Errors
    /// [`TransportedRefusal::NotTransported`] for a source in the model's own
    /// population, and every refusal of [`Self::declared_point`].
    pub fn from_shifted_cohort(source: &CohortDecisionSource) -> Result<Self, TransportedRefusal> {
        let Some(shift) = source.shift() else {
            return Err(TransportedRefusal::NotTransported);
        };
        let mut shift_identity = CanonicalWriter::new(TAG_SHIFT);
        shift_identity
            .text(&shift.source_population)
            .text(&shift.target_population)
            .texts(&shift.invariant_modifiers)
            .text(&shift.assumption);
        let mean = cohort_mean_source(source);
        let mut assumptions = vec![shift.assumption.clone()];
        assumptions
            .extend(shift.invariant_modifiers.iter().map(|m| format!("invariant_modifier:{m}")));
        let declaration = TransportDeclaration {
            route: TransportedRoute::ShiftedCohort,
            source_identity: source.identity().to_owned(),
            source_populations: vec![shift.source_population.clone()],
            target_population: shift.target_population.clone(),
            transport_identities: vec![shift_identity.finish_hex()],
            assumptions,
            support: source.support_status(),
            uncalibrated_standard_error: source
                .uncalibrated_standard_error()
                .map(|error| error.value),
            provider_id: mean.provider_id.clone(),
            snapshot_id: mean.snapshot_id.clone(),
            causal_contract_id: mean.causal_contract_id.clone(),
        };
        Self::build_point(mean.coordinates, mean.means, &declaration)
    }

    /// The target-population value of a transported policy evaluation, stated
    /// at `coordinate`.
    ///
    /// # Errors
    /// Every refusal of [`Self::declared_point`].
    pub fn from_transported_policy(
        evaluation: &TransportedPolicyEvaluation,
        coordinate: ScientificQuantity,
        provider_id: &str,
    ) -> Result<Self, TransportedRefusal> {
        let premises = &evaluation.premises;
        let mut assumptions = vec![
            premises.selection.to_owned(),
            premises.invariance.to_owned(),
            premises.overlap.to_owned(),
        ];
        assumptions
            .extend(evaluation.shifted_mechanisms.iter().map(|m| format!("shifted:{}", m.name())));
        let declaration = TransportDeclaration {
            route: TransportedRoute::Policy,
            source_identity: evaluation.identity.clone(),
            source_populations: vec![evaluation.source_population.clone()],
            target_population: evaluation.target_population.clone(),
            transport_identities: vec![evaluation.diagram_identity.clone()],
            assumptions,
            support: SupportStatus::Extrapolative,
            uncalibrated_standard_error: None,
            provider_id: provider_id.to_owned(),
            snapshot_id: evaluation.identity.clone(),
            causal_contract_id: evaluation.diagram_identity.clone(),
        };
        Self::build_point(vec![coordinate], vec![evaluation.target_value], &declaration)
    }

    /// The value of a synthesis that contains transported claims, stated at
    /// `coordinate`. The synthesis's model variance is retained as an
    /// uncalibrated standard error and is not inherited by a decision.
    ///
    /// # Errors
    /// [`TransportedRefusal::NotTransported`] for a synthesis of native
    /// sources only, and every refusal of [`Self::declared_point`]. The source
    /// populations and the target are the ones the ledger's transport
    /// identities recorded, never caller text; a coordinate stated in another
    /// population is [`TransportedRefusal::QuantityNotInTarget`].
    pub fn from_transported_synthesis(
        synthesis: &TransportedSynthesis,
        coordinate: ScientificQuantity,
        provider_id: &str,
    ) -> Result<Self, TransportedRefusal> {
        let Some(target) = synthesis.target_population.clone() else {
            return Err(TransportedRefusal::NotTransported);
        };
        if synthesis.transported_sources.is_empty() || synthesis.transport_identities.is_empty() {
            return Err(TransportedRefusal::NotTransported);
        }
        let source_populations = synthesis.source_populations.clone();
        let result = synthesis.result.to_hex();
        let declaration = TransportDeclaration {
            route: TransportedRoute::Synthesis,
            source_identity: result.clone(),
            source_populations,
            target_population: target,
            transport_identities: synthesis
                .transport_identities
                .iter()
                .map(antecedent_core::SemanticDigest::to_hex)
                .collect(),
            assumptions: vec![
                format!("synthesis_method:{}", synthesis.method),
                "transport_error_counted_once_per_transport".to_owned(),
            ],
            support: SupportStatus::Extrapolative,
            uncalibrated_standard_error: (synthesis.variance.is_finite()
                && synthesis.variance >= 0.0)
                .then(|| synthesis.variance.sqrt()),
            provider_id: provider_id.to_owned(),
            snapshot_id: result.clone(),
            causal_contract_id: result,
        };
        Self::build_point(vec![coordinate], vec![synthesis.estimate], &declaration)
    }

    /// Merge transported points that were carried by one transport into one
    /// grid of coordinates, so one decision or inverse query can read them
    /// together (for example one transported value per candidate policy).
    ///
    /// Every response must be a point with the same route, populations,
    /// transport identities, assumptions and support label; coordinates must
    /// be distinct. The merged source identity binds every member's identity,
    /// and no member's standard error is carried over.
    ///
    /// # Errors
    /// [`TransportedRefusal::InvalidValue`] for no response or a law member;
    /// [`TransportedRefusal::MixedTransports`] for responses carried by
    /// different transports; the refusals of [`Self::declared_point`].
    pub fn merge_points(responses: &[Self]) -> Result<Self, TransportedRefusal> {
        let Some(first) = responses.first() else {
            return Err(TransportedRefusal::InvalidValue("at least one response"));
        };
        let mut coordinates = Vec::new();
        let mut means = Vec::new();
        let mut identities: Vec<&str> = Vec::new();
        for response in responses {
            let TransportedValue::Means(mean) = &response.value else {
                return Err(TransportedRefusal::InvalidValue("only transported points merge"));
            };
            let (a, b) = (&response.receipt, &first.receipt);
            if a.route != b.route
                || a.source_populations != b.source_populations
                || a.target_population != b.target_population
                || a.transport_identities != b.transport_identities
                || a.assumptions != b.assumptions
                || a.support != b.support
            {
                return Err(TransportedRefusal::MixedTransports);
            }
            coordinates.extend(mean.coordinates.iter().cloned());
            means.extend(mean.means.iter().copied());
            identities.push(a.source_identity.as_str());
        }
        identities.sort_unstable();
        let mut writer = CanonicalWriter::new(TAG_MERGE);
        writer.texts(&identities);
        let merged = writer.finish_hex();
        let TransportedValue::Means(head) = &first.value else {
            return Err(TransportedRefusal::InvalidValue("only transported points merge"));
        };
        let declaration = TransportDeclaration {
            route: first.receipt.route,
            source_identity: merged.clone(),
            source_populations: first.receipt.source_populations.clone(),
            target_population: first.receipt.target_population.clone(),
            transport_identities: first.receipt.transport_identities.clone(),
            assumptions: first.receipt.assumptions.clone(),
            support: first.receipt.support_status().ok_or_else(|| {
                TransportedRefusal::SupportLabelInflated(first.receipt.support.clone())
            })?,
            uncalibrated_standard_error: None,
            provider_id: head.provider_id.clone(),
            snapshot_id: merged.clone(),
            causal_contract_id: merged,
        };
        Self::build_point(coordinates, means, &declaration)
    }

    /// Rebuild a transported point from a stored receipt and mean grid, for a
    /// consumer that re-derives a decision from an artifact. The receipt is
    /// validated exactly as at construction; nothing in it is trusted.
    ///
    /// # Errors
    /// A law-valued receipt, an invalid value, an invalid receipt or a
    /// coordinate not stated in the receipt's target population.
    pub(crate) fn from_stored_point(
        mean: MeanSource,
        receipt: TransportReceipt,
    ) -> Result<Self, TransportedRefusal> {
        if receipt.representation != TransportedRepresentation::PointOnly {
            return Err(TransportedRefusal::InvalidValue("a stored point carries a point receipt"));
        }
        if mean.coordinates.is_empty()
            || mean.coordinates.len() != mean.means.len()
            || mean.means.iter().any(|m| !m.is_finite())
        {
            return Err(TransportedRefusal::InvalidValue("one finite mean per coordinate"));
        }
        for (position, coordinate) in mean.coordinates.iter().enumerate() {
            if mean.coordinates[..position]
                .iter()
                .any(|earlier| earlier.require_same_coordinate(coordinate).is_ok())
            {
                return Err(TransportedRefusal::InvalidValue("coordinates are distinct"));
            }
        }
        receipt.validate()?;
        check_in_target(&mean.coordinates, &receipt.target_population)?;
        Ok(Self { value: TransportedValue::Means(mean), receipt })
    }

    /// The value.
    #[must_use]
    pub const fn value(&self) -> &TransportedValue {
        &self.value
    }

    /// The transport identity and labels.
    #[must_use]
    pub const fn receipt(&self) -> &TransportReceipt {
        &self.receipt
    }

    /// What the value is.
    #[must_use]
    pub const fn representation(&self) -> TransportedRepresentation {
        self.receipt.representation
    }

    /// The coordinates the value answers, in value order.
    #[must_use]
    pub fn coordinates(&self) -> Vec<ScientificQuantity> {
        match &self.value {
            TransportedValue::Means(mean) => mean.coordinates.clone(),
            TransportedValue::Law(law) => law
                .quantities()
                .iter()
                .filter_map(|wire| ScientificQuantity::try_from(wire.clone()).ok())
                .collect(),
        }
    }
}

fn check_in_target(
    coordinates: &[ScientificQuantity],
    target: &str,
) -> Result<(), TransportedRefusal> {
    match coordinates.iter().find(|c| c.population_id != target) {
        Some(coordinate) => Err(TransportedRefusal::QuantityNotInTarget {
            coordinate: coordinate.variable_id.clone(),
            expected: target.to_owned(),
            supplied: coordinate.population_id.clone(),
        }),
        None => Ok(()),
    }
}

/// The routes with a typed constructor cannot be declared by hand: their
/// populations and identities come from the producing result.
fn require_declarable_route(route: TransportedRoute) -> Result<(), TransportedRefusal> {
    match route {
        TransportedRoute::Policy
        | TransportedRoute::ShiftedCohort
        | TransportedRoute::Synthesis => Err(TransportedRefusal::RouteNeedsTypedSource(route)),
        TransportedRoute::CohortEffect
        | TransportedRoute::Mechanism
        | TransportedRoute::Declared => Ok(()),
    }
}

/// Why a transported response was refused as a decision source.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum TransportedRefusal {
    /// The transport identity is incomplete.
    TransportIdentityIncomplete,
    /// A typed route's transport identity is not a 32-byte digest.
    IdentityNotDigest,
    /// The route has a typed constructor and cannot be declared by hand.
    RouteNeedsTypedSource(TransportedRoute),
    /// The object offered was not carried to another population.
    NotTransported,
    /// A coordinate is not stated in the target population.
    QuantityNotInTarget {
        /// Variable identity of the coordinate.
        coordinate: String,
        /// The target population.
        expected: String,
        /// The population the coordinate names.
        supplied: String,
    },
    /// The support label is unknown or stronger than `extrapolative`.
    SupportLabelInflated(String),
    /// A transported law is labelled exact or measured.
    CalibrationLabelInflated,
    /// A transported law is labelled native.
    NativeLabelOnTransported,
    /// A malformed value.
    InvalidValue(&'static str),
    /// The request needs a law and the transported value is a point.
    LawNotSupplied {
        /// What in the request needs the law.
        needed: &'static str,
    },
    /// The request needs an interval.
    IntervalNotLicensed,
    /// Responses merged into one grid were not carried by one transport.
    MixedTransports,
    /// The policy's weakest admitted support is stronger than the response's.
    SupportBelowPolicy {
        /// The response's label.
        supplied: SupportStatus,
        /// The policy's weakest admitted label.
        required: SupportStatus,
    },
    /// The decision boundary refused.
    Boundary(BoundaryError),
}

impl TransportedRefusal {
    /// Registered reason code.
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            Self::TransportIdentityIncomplete => {
                antecedent_core::reason_code!("transport_missing_evidence")
            }
            Self::IdentityNotDigest => antecedent_core::reason_code!("transport_missing_evidence"),
            Self::NotTransported
            | Self::RouteNeedsTypedSource(_)
            | Self::InvalidValue(_) => {
                antecedent_core::reason_code!("invalid_argument")
            }
            Self::QuantityNotInTarget { .. } => {
                antecedent_core::reason_code!("quantity_semantics_mismatch")
            }
            Self::SupportLabelInflated(_)
            | Self::CalibrationLabelInflated
            | Self::NativeLabelOnTransported => {
                antecedent_core::reason_code!("transport_not_certified")
            }
            Self::LawNotSupplied { .. } => {
                antecedent_core::reason_code!("decision_contract_unsatisfied")
            }
            Self::MixedTransports => antecedent_core::reason_code!("transport_not_certified"),
            Self::IntervalNotLicensed => antecedent_core::reason_code!("cell_not_licensed"),
            // The same code every consumer uses to say it does not accept a
            // transported input (inverse queries, cohort forward claims).
            Self::SupportBelowPolicy { .. } => antecedent_core::reason_code!("cell_not_licensed"),
            Self::Boundary(error) => error.reason_code(),
        }
    }

    /// Namespaced detail.
    #[must_use]
    pub fn detail(&self) -> String {
        match self {
            Self::TransportIdentityIncomplete => DETAIL_IDENTITY_INCOMPLETE.to_owned(),
            Self::IdentityNotDigest => DETAIL_IDENTITY_NOT_DIGEST.to_owned(),
            Self::RouteNeedsTypedSource(_) => DETAIL_ROUTE_NEEDS_TYPED_SOURCE.to_owned(),
            Self::NotTransported => DETAIL_NOT_TRANSPORTED.to_owned(),
            Self::QuantityNotInTarget { .. } => DETAIL_NOT_IN_TARGET.to_owned(),
            Self::SupportLabelInflated(_) => DETAIL_SUPPORT_INFLATED.to_owned(),
            Self::CalibrationLabelInflated => DETAIL_CALIBRATION_INFLATED.to_owned(),
            Self::NativeLabelOnTransported => DETAIL_NATIVE_LABEL.to_owned(),
            Self::InvalidValue(_) => DETAIL_INVALID_VALUE.to_owned(),
            Self::LawNotSupplied { .. } => DETAIL_LAW_NOT_SUPPLIED.to_owned(),
            Self::IntervalNotLicensed => DETAIL_INTERVAL_NOT_LICENSED.to_owned(),
            Self::MixedTransports => DETAIL_MIXED_TRANSPORTS.to_owned(),
            Self::SupportBelowPolicy { .. } => DETAIL_SUPPORT_BELOW_POLICY.to_owned(),
            Self::Boundary(error) => error.to_refusal().detail,
        }
    }

    /// Structured refusal with a stated remedy.
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let build = |detail: &str| RefusalBuilder::new(self.code(), STAGE, detail);
        match self {
            Self::TransportIdentityIncomplete => build(DETAIL_IDENTITY_INCOMPLETE)
                .remedy(
                    "name the source populations (none of them the target), the target \
                     population, the selection diagram and the assumptions the value was carried under",
                )
                .build(),
            Self::IdentityNotDigest => build(DETAIL_IDENTITY_NOT_DIGEST)
                .remedy("a typed route's transport identity is the canonical digest its producing route computed; pass that digest, or declare the route as `declared`")
                .build(),
            Self::RouteNeedsTypedSource(route) => build(DETAIL_ROUTE_NEEDS_TYPED_SOURCE)
                .offending(route.as_str())
                .remedy("build the response from the producing result (`from_shifted_cohort`, `from_transported_policy`, `from_transported_synthesis`) so its populations and identities are copied, not typed")
                .build(),
            Self::NotTransported => build(DETAIL_NOT_TRANSPORTED)
                .remedy(
                    "a value in the model's own population is a native source; decide on it \
                     directly, or declare the shift or selection diagram that carries it",
                )
                .build(),
            Self::QuantityNotInTarget { coordinate, expected, supplied } => {
                build(DETAIL_NOT_IN_TARGET)
                    .offending(coordinate.clone())
                    .expected(expected.clone())
                    .supplied(supplied.clone())
                    .remedy("state the transported value in the target population of its transport identity")
                    .build()
            }
            Self::SupportLabelInflated(label) => build(DETAIL_SUPPORT_INFLATED)
                .supplied(label.clone())
                .remedy("a transported value is labelled extrapolative at best; transport never raises a support label")
                .build(),
            Self::CalibrationLabelInflated => build(DETAIL_CALIBRATION_INFLATED)
                .remedy("a transported law carries no measured calibration; relabel it point-only or unmeasured")
                .build(),
            Self::NativeLabelOnTransported => build(DETAIL_NATIVE_LABEL)
                .remedy("a transported law is an external or derived claim, never native target data")
                .build(),
            Self::InvalidValue(what) => {
                build(DETAIL_INVALID_VALUE).offending(*what).build()
            }
            Self::LawNotSupplied { needed } => build(DETAIL_LAW_NOT_SUPPLIED)
                .offending(*needed)
                .remedy(
                    "a transported mean is one expectation; supply aligned outcome draws \
                     under a declared transport model for a probability, quantile, nonlinear \
                     utility, hard constraint or regret",
                )
                .build(),
            Self::MixedTransports => build(DETAIL_MIXED_TRANSPORTS)
                .remedy("merge only values carried by one transport: the same populations, selection diagram, assumptions and support label")
                .build(),
            Self::IntervalNotLicensed => build(DETAIL_INTERVAL_NOT_LICENSED)
                .remedy(
                    "none available: the transport step and the support check have no \
                     measured coverage; the value is point-only under the declared premises",
                )
                .build(),
            Self::SupportBelowPolicy { supplied, required } => {
                build(DETAIL_SUPPORT_BELOW_POLICY)
                    .expected(required.as_str())
                    .supplied(supplied.as_str())
                    .remedy("accept transported evidence explicitly by naming a weakest admitted support of extrapolative")
                    .build()
            }
            Self::Boundary(error) => at_stage_of(error),
        }
    }
}

fn at_stage_of(error: &BoundaryError) -> ExternalRefusal {
    crate::signal::at_stage(error.to_refusal(), STAGE)
}

impl std::fmt::Display for TransportedRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}: {}", antecedent_core::reason_code::PREFIX, self.code(), self.detail())
    }
}

impl std::error::Error for TransportedRefusal {}

/// What a decision on a transported response asks of it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransportedDemand {
    /// The expectation of an affine utility (the contract decides whether more
    /// is needed).
    ExpectedUtility,
    /// A law: a probability, quantile or nonlinear functional.
    Law,
    /// An interval, band or posterior for the value.
    Interval,
}

/// The first thing in `contract` that a mean cannot answer, when there is one.
#[must_use]
pub fn contract_needs_law(contract: &DecisionContract) -> Option<&'static str> {
    if !matches!(
        contract.criterion,
        DecisionCriterion::PosteriorExpectedUtility | DecisionCriterion::PosteriorExpectedLoss
    ) {
        return Some("criterion");
    }
    if !contract.constraints.is_empty() {
        return Some("hard_constraint");
    }
    for action in &contract.actions {
        if !action.utility.is_affine() {
            return Some("nonlinear_utility");
        }
        if action.inputs.iter().any(|q| q.functional_id == "outcome") {
            return Some("outcome_law_input");
        }
    }
    None
}

/// A decision made on a transported response.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportedDecision {
    /// The decision at the boundary, per action.
    pub decision: SupportedDecision,
    /// The transport identity of the value the decision stands on.
    pub receipt: TransportReceipt,
    /// Assumptions of the decision followed by the transport's.
    pub assumptions: Vec<String>,
    /// Canonical identity of the decision claim: contract, transport receipt
    /// and the lineage of the input.
    pub identity: String,
    /// Always `point_only`.
    pub inference: &'static str,
}

/// Evaluate `contract` on a transported response.
///
/// The request is checked before any number is read: an interval is refused for
/// every transported value, and a law (demanded, or needed by the contract's
/// criterion, constraints or utilities) is refused for a point. The policy must
/// admit the response's support label. The decision itself is the boundary's
/// own [`evaluate_with_support`], so the answered actions, their values and the
/// verdict are exactly what a native source of the same value would give; what
/// differs is that the result carries the transport identity and labels.
///
/// # Errors
/// [`TransportedRefusal::IntervalNotLicensed`],
/// [`TransportedRefusal::LawNotSupplied`],
/// [`TransportedRefusal::SupportBelowPolicy`], and any refusal of the boundary
/// (an invalid contract, a missing coordinate under `RequireAllActions`, a
/// non-joint alignment).
pub fn evaluate_transported_decision(
    contract: &DecisionContract,
    response: &TransportedResponse,
    demand: TransportedDemand,
    policy: &SupportPolicy,
) -> Result<TransportedDecision, TransportedRefusal> {
    if demand == TransportedDemand::Interval {
        return Err(TransportedRefusal::IntervalNotLicensed);
    }
    let needed = match (demand, contract_needs_law(contract)) {
        (TransportedDemand::Law, _) => Some("law_request"),
        (_, other) => other,
    };
    if let (Some(needed), TransportedValue::Means(_)) = (needed, response.value()) {
        return Err(TransportedRefusal::LawNotSupplied { needed });
    }
    let supplied = response.receipt.support_status().ok_or_else(|| {
        TransportedRefusal::SupportLabelInflated(response.receipt.support.clone())
    })?;
    if supplied.severity() > policy.weakest_support.severity() {
        return Err(TransportedRefusal::SupportBelowPolicy {
            supplied,
            required: policy.weakest_support,
        });
    }
    let id = format!("transported:{}", response.receipt.source_identity);
    let input = match &response.value {
        TransportedValue::Means(mean) => {
            let statuses: Vec<SupportStatus> = vec![supplied; mean.coordinates.len()];
            DecisionInput::from_mean_source(
                &id,
                mean.clone(),
                &statuses,
                &TrustEvidence::None,
                TrustRequirement::Unrestricted,
            )
        }
        // The receipt's label governs every coordinate of a law as it does
        // for a point; the artifact's mask can only weaken it further.
        TransportedValue::Law(law) => DecisionInput::from_distribution_artifact(
            &id,
            law,
            &TrustEvidence::None,
            TrustRequirement::Unrestricted,
        )
        .map(|input| input.with_support_floor(supplied)),
    }
    .map_err(TransportedRefusal::Boundary)?;
    let lineage = input.provenance().lineage_digest.clone();
    let decision = evaluate_with_support(contract, std::slice::from_ref(&input), policy)
        .map_err(TransportedRefusal::Boundary)?;
    let mut assumptions = decision.assumptions.clone();
    assumptions.push(format!(
        "the value was carried from {} to {} by route {} under declared assumptions; it is not native target data",
        response.receipt.source_populations.join(","),
        response.receipt.target_population,
        response.receipt.route.as_str(),
    ));
    assumptions.extend(response.receipt.assumptions.iter().cloned());
    assumptions.push(
        "the decision is point-only: the transport step carries no measured coverage".to_owned(),
    );
    let mut writer = CanonicalWriter::new(TAG_DECISION);
    writer
        .word(u64::from(TRANSPORTED_RESPONSE_VERSION))
        .text(&decision.contract_identity)
        .text(&response.receipt.canonical_identity())
        .text(&lineage);
    Ok(TransportedDecision {
        decision,
        receipt: response.receipt.clone(),
        assumptions,
        identity: writer.finish_hex(),
        inference: TRANSPORTED_RESPONSE_INFERENCE,
    })
}
