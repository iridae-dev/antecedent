//! Revalidation of transported claims after a refresh or a state change.
//!
//! A transported claim is a claim about a target population, derived from source
//! evidence under a selection diagram, possibly carrying a calibration record
//! established at one target. When the target, a source's evidence, the diagram or
//! the support changes, every transported claim and every decision resting on one
//! is classified `still_valid`, `recompute` or `withdrawn`, in transport terms
//! ([`TransportReason`]) and through the one claim-level graph of
//! [`super::composition_verification`].
//!
//! * [`TransportSceneBuilder`] declares a scene from the facade's own transported
//!   objects: each [`TransportedResponse`] contributes its receipt (source
//!   populations, target population, selection-diagram identities, support
//!   label), its value digest and the calibration it states.
//! * [`revalidate`] classifies a scene against the earlier one through the
//!   contract's `revalidate_transport` and seals the same two graphs as a
//!   `composition_verification_v1` artifact, so the classification exports and a
//!   fresh consumer replays it ([`super::composition_verification`]).
//! * [`TransportClaimState`] holds the current scene and advances it only when a
//!   revalidation succeeds, so a refused refresh leaves the state unchanged.
//!
//! # The rule this runs
//!
//! Transported calibration is established at one target. A record held at one
//! target is never read as holding at another: a claim whose record was
//! established at a different target (or for a different quantity) than the one it
//! now answers is withdrawn for calibration with
//! [`TransportReason::CalibrationFromAnotherTarget`], and its point value is left
//! as it was. A claim that inherits a record from another claim holds it only
//! while that claim holds it at the same target and quantity. No code here
//! declares or measures a calibration record.
//!
//! # Lifecycle
//!
//! * **Refresh with outcome-only data of a source population** changes that
//!   source's evidence identity: every claim counting the source **recomputes**
//!   (the earlier fit and uncertainty carry no authority); the declared selection
//!   diagram, structure and quantity are **kept**; a claim about another target
//!   is **refused** its calibration rather than inheriting it.
//! * **A change of target population or diagram** recomputes every transported
//!   claim; a target outside the support withdraws them.
//! * **State change.** [`invalidation_effects_for_claims`] maps entries of the
//!   state's invalidation log onto claim keys, so a stale query result recomputes
//!   its claim and a withdrawn premise withdraws it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use antecedent_contract::ExternalRefusal;
use antecedent_core::{CanonicalWriter, SemanticDigest, StateVersion, SupportStatus};
use antecedent_state::{InvalidationLog, InvalidationTarget};

pub use antecedent_contract::transport_revalidation::{
    SceneCalibration, SceneClaim, SceneClaimKind, SceneSource, SceneTarget,
    TRANSPORT_REVALIDATION_VERSION, TransportClaimRevalidation, TransportReason,
    TransportRevalidation, TransportScene, revalidate_transport, transport_coordinate,
};

use antecedent_design::decision_transported_source::{TransportedResponse, TransportedValue};
use antecedent_design::inverse_forward_source::forward_claim_digest;
use antecedent_design::inverse_query::ForwardClaim;

use super::composition_verification::{
    CalibrationState, CompositionVerification, CompositionVerificationRefusal, GraphError,
    ObservedEffect, PriorAuthority, Revalidation, SupportStanding, TrustStanding,
    invalidation_effects,
};

const TAG_CLAIM: &str = "antecedent.transport_revalidation.facade.claim";
const TAG_QUANTITY: &str = "antecedent.transport_revalidation.facade.quantity";
const TAG_STRUCTURE: &str = "antecedent.transport_revalidation.facade.structure";
const TAG_DIAGRAM: &str = "antecedent.transport_revalidation.facade.diagram";
const TAG_TARGET: &str = "antecedent.transport_revalidation.facade.target";

/// Detail: the scene's graph does not build.
pub const DETAIL_SCENE_INVALID: &str = "transport_revalidation.scene_invalid";
/// Detail: a claim names a target other than the scene's.
pub const DETAIL_TARGET_MISMATCH: &str = "transport_revalidation.claim_target_mismatch";
/// Detail: a claim counts a source population the scene does not declare.
pub const DETAIL_SOURCE_UNDECLARED: &str = "transport_revalidation.source_undeclared";
/// Detail: a claim key was used twice.
pub const DETAIL_DUPLICATE_CLAIM: &str = "transport_revalidation.duplicate_claim";
/// Detail: no claim has the key.
pub const DETAIL_UNKNOWN_CLAIM: &str = "transport_revalidation.unknown_claim";
/// Detail: the claim's value digest could not be formed.
pub const DETAIL_VALUE_DIGEST: &str = "transport_revalidation.value_digest";

/// One refusal as plain data.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransportRevalidationRefusal {
    /// Registered runtime reason code.
    pub code: &'static str,
    /// Namespaced `transport_revalidation.*` (or `composition_verification.*`) detail.
    pub detail: String,
    /// What was refused.
    pub message: String,
    /// What the caller can change to proceed.
    pub remedy: Option<&'static str>,
}

impl std::fmt::Display for TransportRevalidationRefusal {
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

impl std::error::Error for TransportRevalidationRefusal {}

impl TransportRevalidationRefusal {
    fn new(
        code: &'static str,
        detail: &str,
        message: impl Into<String>,
        remedy: &'static str,
    ) -> Self {
        Self { code, detail: detail.to_owned(), message: message.into(), remedy: Some(remedy) }
    }

    fn graph(error: &GraphError) -> Self {
        Self::new(
            antecedent_core::reason_code!("invalid_argument"),
            DETAIL_SCENE_INVALID,
            error.to_string(),
            "declare unique keys, dependencies between declared claims without a cycle, and a \
             calibration dependency for every inherited calibration",
        )
    }

    fn verification(error: CompositionVerificationRefusal) -> Self {
        Self { code: error.code, detail: error.detail, message: error.message, remedy: None }
    }
}

fn digest_of(writer: &CanonicalWriter) -> SemanticDigest {
    SemanticDigest::from_bytes(writer.finish())
}

/// The graph key of a claim, as used by observed effects.
#[must_use]
pub fn claim_node_key(key: &str) -> String {
    format!("claim:{key}")
}

/// The graph key of a source population, as used by observed effects.
#[must_use]
pub fn source_node_key(population: &str) -> String {
    format!("source:{population}")
}

/// The graph key of the target population.
pub const TARGET_NODE_KEY: &str = "target_population";
/// The graph key of the selection diagram.
pub const DIAGRAM_NODE_KEY: &str = "selection_diagram";

/// Declares a [`TransportScene`] from transported responses.
#[derive(Clone, Debug)]
pub struct TransportSceneBuilder {
    target_population: String,
    target_data: SemanticDigest,
    target_support: SupportStanding,
    sources: BTreeMap<String, SemanticDigest>,
    claims: Vec<SceneClaim>,
    diagrams: BTreeSet<String>,
    refusal: Option<TransportRevalidationRefusal>,
}

impl TransportSceneBuilder {
    /// A scene about `target_population`; `target_data_identity` is the content
    /// identity of the target's declaration and data.
    #[must_use]
    pub fn new(target_population: impl Into<String>, target_data_identity: SemanticDigest) -> Self {
        Self {
            target_population: target_population.into(),
            target_data: target_data_identity,
            target_support: SupportStanding::Supported,
            sources: BTreeMap::new(),
            claims: Vec::new(),
            diagrams: BTreeSet::new(),
            refusal: None,
        }
    }

    /// The identity the scene gives its target: the population and the content
    /// identity of its declaration and data. A calibration record established at
    /// this target is declared with it ([`SceneCalibration::Held`]).
    #[must_use]
    pub fn target_identity(&self) -> SemanticDigest {
        let mut target = CanonicalWriter::new(TAG_TARGET);
        target.text(&self.target_population).bytes(self.target_data.as_bytes());
        digest_of(&target)
    }

    /// Whether the target lies inside the support the transport needs.
    #[must_use]
    pub fn target_support(mut self, support: SupportStanding) -> Self {
        self.target_support = support;
        self
    }

    /// Declare a source population with the content identity of its evidence.
    #[must_use]
    pub fn source(
        mut self,
        population: impl Into<String>,
        evidence_identity: SemanticDigest,
    ) -> Self {
        self.sources.insert(population.into(), evidence_identity);
        self
    }

    /// Add a transported claim from a response and the calibration it states.
    ///
    /// The claim's identity binds the receipt and the value; its structure is
    /// the receipt's selection-diagram identities; its sources are the receipt's
    /// source populations; its support standing follows the receipt's label.
    #[must_use]
    pub fn transported(
        mut self,
        key: impl Into<String>,
        response: &TransportedResponse,
        calibration: SceneCalibration,
        trust: TrustStanding,
    ) -> Self {
        if self.refusal.is_some() {
            return self;
        }
        let key = key.into();
        if self.claims.iter().any(|c| c.key == key) {
            self.refusal = Some(TransportRevalidationRefusal::new(
                antecedent_core::reason_code!("invalid_argument"),
                DETAIL_DUPLICATE_CLAIM,
                format!("claim `{key}` is declared twice"),
                "give every claim of the scene a distinct key",
            ));
            return self;
        }
        let receipt = response.receipt();
        if receipt.target_population != self.target_population {
            self.refusal = Some(TransportRevalidationRefusal::new(
                antecedent_core::reason_code!("quantity_semantics_mismatch"),
                DETAIL_TARGET_MISMATCH,
                format!(
                    "claim `{key}` is about `{}`, the scene's target is `{}`",
                    receipt.target_population, self.target_population
                ),
                "build one scene per target population; a claim about another target is not \
                 revalidated against this one",
            ));
            return self;
        }
        let claim_value = match response.value() {
            TransportedValue::Means(mean) => ForwardClaim::Means(mean.clone()),
            TransportedValue::Law(law) => ForwardClaim::Law(Box::new(law.clone())),
        };
        let value_digest = match forward_claim_digest(&claim_value) {
            Ok(digest) => digest,
            Err(error) => {
                self.refusal = Some(TransportRevalidationRefusal::new(
                    antecedent_core::reason_code!("invalid_argument"),
                    DETAIL_VALUE_DIGEST,
                    error.to_string(),
                    "supply a value whose metadata encodes",
                ));
                return self;
            }
        };
        let mut identity = CanonicalWriter::new(TAG_CLAIM);
        identity.text(&receipt.canonical_identity()).text(&value_digest);
        let mut quantity = CanonicalWriter::new(TAG_QUANTITY);
        let coordinates = response.coordinates();
        quantity.word(coordinates.len() as u64);
        for coordinate in &coordinates {
            coordinate.write_canonical(&mut quantity);
        }
        let mut sorted_diagrams = receipt.transport_identities.clone();
        sorted_diagrams.sort_unstable();
        sorted_diagrams.dedup();
        let mut structure = CanonicalWriter::new(TAG_STRUCTURE);
        structure.texts(&sorted_diagrams);
        self.diagrams.extend(sorted_diagrams);
        let mut claim = SceneClaim::new(
            key,
            SceneClaimKind::Transported,
            digest_of(&identity),
            digest_of(&quantity),
        );
        claim.structure = Some(digest_of(&structure));
        claim.sources = receipt.source_populations.clone();
        claim.trust = trust;
        claim.calibration = calibration;
        if let Some(status) = receipt.support_status() {
            if status.severity() > SupportStatus::Extrapolative.severity() {
                claim.support = SupportStanding::Unsupported {
                    coordinates: {
                        let mut names: Vec<String> =
                            coordinates.iter().map(|c| c.variable_id.clone()).collect();
                        names.sort_unstable();
                        names.dedup();
                        names
                    },
                };
            }
        }
        self.claims.push(claim);
        self
    }

    /// Add a decision resting on claims of the scene.
    #[must_use]
    pub fn decision(
        mut self,
        key: impl Into<String>,
        identity: SemanticDigest,
        quantity: SemanticDigest,
        rests_on: &[&str],
    ) -> Self {
        if self.refusal.is_some() {
            return self;
        }
        let mut claim = SceneClaim::new(key, SceneClaimKind::Decision, identity, quantity);
        claim.rests_on = rests_on.iter().map(|k| (*k).to_owned()).collect();
        self.claims.push(claim);
        self
    }

    /// The scene.
    ///
    /// # Errors
    /// A claim about another target, a repeated key, a claim counting a source
    /// population the scene did not declare, or a scene whose graph does not
    /// build (an unknown or cyclic dependency).
    pub fn build(self) -> Result<TransportScene, TransportRevalidationRefusal> {
        if let Some(refusal) = self.refusal {
            return Err(refusal);
        }
        for claim in &self.claims {
            if let Some(missing) = claim.sources.iter().find(|s| !self.sources.contains_key(*s)) {
                return Err(TransportRevalidationRefusal::new(
                    antecedent_core::reason_code!("transport_missing_evidence"),
                    DETAIL_SOURCE_UNDECLARED,
                    format!(
                        "claim `{}` counts source population `{missing}`, which the scene does \
                         not declare",
                        claim.key
                    ),
                    "declare the source population with the identity of its evidence",
                ));
            }
        }
        let target = self.target_identity();
        let mut diagram = CanonicalWriter::new(TAG_DIAGRAM);
        diagram.texts(&self.diagrams.into_iter().collect::<Vec<_>>());
        let scene = TransportScene {
            target: SceneTarget { identity: target, support: self.target_support },
            selection_diagram: digest_of(&diagram),
            sources: self
                .sources
                .into_iter()
                .map(|(key, identity)| SceneSource { key, identity })
                .collect(),
            claims: self.claims,
        };
        scene.graph().map_err(|e| TransportRevalidationRefusal::graph(&e))?;
        Ok(scene)
    }
}

/// Map entries of the state's invalidation log onto claim keys.
///
/// A claim bound to a query result or model the log invalidated after `since`
/// recomputes; one whose premise the log shows removed is withdrawn. The returned
/// effects are keyed by graph node (`claim:<key>`) as [`revalidate`] expects.
#[must_use]
pub fn invalidation_effects_for_claims(
    log: &InvalidationLog,
    since: StateVersion,
    bindings: &[(&str, InvalidationTarget)],
) -> BTreeMap<String, ObservedEffect> {
    let keys: Vec<String> = bindings.iter().map(|(key, _)| claim_node_key(key)).collect();
    let nodes: Vec<(&str, InvalidationTarget)> = keys
        .iter()
        .zip(bindings)
        .map(|(node, (_, target))| (node.as_str(), target.clone()))
        .collect();
    invalidation_effects(log, since, &nodes)
}

/// One claim's classification as plain data.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransportClaimInspection {
    /// The claim key.
    pub key: String,
    /// `transported` or `decision`.
    pub kind: &'static str,
    /// `still_valid`, `recompute` or `withdrawn`.
    pub class: &'static str,
    /// Why, in transport terms (empty only for a `still_valid` claim).
    pub reasons: Vec<&'static str>,
    /// What the caller can do about each reason.
    pub remedies: Vec<&'static str>,
    /// `retained` or `not_retained`.
    pub prior_authority: &'static str,
    /// The claim's uncertainty guarantee after revalidation.
    pub calibration: &'static str,
    /// Whether the calibration was withdrawn (the point value is left as it was).
    pub calibration_withdrawn: bool,
}

/// What an inspection of a revalidation shows.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransportRevalidationInspection {
    /// Every claim, in key order.
    pub claims: Vec<TransportClaimInspection>,
    /// Claims in the earlier scene and absent now.
    pub removed: Vec<String>,
    /// Identity of the classification (hex).
    pub identity: String,
}

/// A scene's classification against the earlier scene.
#[derive(Clone, Debug)]
pub struct TransportRevalidationReport {
    revalidation: TransportRevalidation,
    verification: CompositionVerification,
}

/// Classify every claim of `current` against `previous` (a first verification
/// when `None`) under the observed `effects`.
///
/// # Errors
/// A scene whose graph does not build or validate, or the refusal of the
/// composition verification (a bound exceeded).
pub fn revalidate(
    previous: Option<&TransportScene>,
    current: &TransportScene,
    effects: &BTreeMap<String, ObservedEffect>,
) -> Result<TransportRevalidationReport, TransportRevalidationRefusal> {
    let before = previous.unwrap_or(current);
    let revalidation = revalidate_transport(before, current, effects)
        .map_err(|e| TransportRevalidationRefusal::graph(&e))?;
    let previous_graph = match previous {
        Some(scene) => Some(scene.graph().map_err(|e| TransportRevalidationRefusal::graph(&e))?),
        None => None,
    };
    let current_graph = current.graph().map_err(|e| TransportRevalidationRefusal::graph(&e))?;
    let verification =
        CompositionVerification::verify(previous_graph.as_ref(), &current_graph, effects, None)
            .map_err(TransportRevalidationRefusal::verification)?;
    Ok(TransportRevalidationReport { revalidation, verification })
}

impl TransportRevalidationReport {
    /// The classification in transport terms.
    #[must_use]
    pub const fn revalidation(&self) -> &TransportRevalidation {
        &self.revalidation
    }

    /// The sealed claim-level verification of the same two graphs.
    #[must_use]
    pub const fn verification(&self) -> &CompositionVerification {
        &self.verification
    }

    /// The classification of one claim.
    #[must_use]
    pub fn claim(&self, key: &str) -> Option<&TransportClaimRevalidation> {
        self.revalidation.claim(key)
    }

    /// The claim when it is usable as it stands.
    ///
    /// # Errors
    /// The typed refusal of a claim that must be recomputed or whose license was
    /// withdrawn (a calibration record from another target is refused with
    /// `cell_not_licensed`), or `invalid_argument` for an unknown key.
    #[allow(
        clippy::result_large_err,
        reason = "a refusal is the cold path of a once-per-call check; boxing the error would change the public error type"
    )]
    pub fn require_valid(&self, key: &str) -> Result<&TransportClaimRevalidation, ExternalRefusal> {
        let claim = self.revalidation.claim(key).ok_or_else(|| {
            ExternalRefusal::new(
                antecedent_core::reason_code!("invalid_argument"),
                "transport_revalidation",
                DETAIL_UNKNOWN_CLAIM,
            )
            .with_offending(key.to_owned())
        })?;
        match claim.refusal() {
            Some(refusal) => Err(refusal),
            None => Ok(claim),
        }
    }

    /// Identity of the classification (hex).
    #[must_use]
    pub fn identity(&self) -> String {
        self.revalidation.identity().to_hex()
    }

    /// Write the claim-level verification as `composition_verification_v1`; a
    /// fresh consumer replays the classification with
    /// [`super::composition_verification::consume_composition_verification_artifact`].
    ///
    /// # Errors
    /// A blank artifact id or an oversized payload.
    pub fn export(&self, artifact_id: &str) -> Result<Vec<u8>, TransportRevalidationRefusal> {
        self.verification.export(artifact_id).map_err(TransportRevalidationRefusal::verification)
    }

    /// Plain-data summary: every claim's class and its reasons in transport terms.
    #[must_use]
    pub fn inspect(&self) -> TransportRevalidationInspection {
        let claims = self
            .revalidation
            .claims()
            .map(|claim| TransportClaimInspection {
                key: claim.key.clone(),
                kind: match claim.kind {
                    SceneClaimKind::Transported => "transported",
                    SceneClaimKind::Decision => "decision",
                },
                class: match claim.class {
                    Revalidation::StillValid => "still_valid",
                    Revalidation::Recompute => "recompute",
                    Revalidation::Withdrawn => "withdrawn",
                },
                reasons: claim.reasons.iter().map(TransportReason::as_str).collect(),
                remedies: claim.reasons.iter().map(TransportReason::remedy).collect(),
                prior_authority: match claim.authority {
                    PriorAuthority::Retained => "retained",
                    _ => "not_retained",
                },
                calibration: match claim.calibration {
                    CalibrationState::NotApplicable => "not_applicable",
                    CalibrationState::Held => "held",
                    CalibrationState::Inherited => "inherited",
                    CalibrationState::Lost => "lost",
                    CalibrationState::Required => "required",
                    _ => "unknown",
                },
                calibration_withdrawn: claim.calibration_withdrawn(),
            })
            .collect();
        TransportRevalidationInspection {
            claims,
            removed: self.revalidation.removed().to_vec(),
            identity: self.identity(),
        }
    }
}

/// The current scene of a set of transported claims, advanced only by a
/// successful revalidation.
#[derive(Clone, Debug, Default)]
pub struct TransportClaimState {
    current: Option<TransportScene>,
}

impl TransportClaimState {
    /// A state holding no scene: the first observation is a first verification.
    #[must_use]
    pub const fn new() -> Self {
        Self { current: None }
    }

    /// The scene the last successful observation recorded.
    #[must_use]
    pub const fn scene(&self) -> Option<&TransportScene> {
        self.current.as_ref()
    }

    /// Classify `next` against the recorded scene and, only on success, record it.
    /// A refusal leaves the state exactly as it was.
    ///
    /// # Errors
    /// As [`revalidate`].
    pub fn observe(
        &mut self,
        next: TransportScene,
        effects: &BTreeMap<String, ObservedEffect>,
    ) -> Result<TransportRevalidationReport, TransportRevalidationRefusal> {
        let report = revalidate(self.current.as_ref(), &next, effects)?;
        self.current = Some(next);
        Ok(report)
    }
}
