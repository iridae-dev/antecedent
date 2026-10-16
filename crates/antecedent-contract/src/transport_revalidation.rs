//! Revalidation of transported claims after a change of target population, source
//! evidence, selection diagram or support.
//!
//! A transported claim is a claim about a target population, derived from source
//! evidence under a selection diagram, and possibly carrying a calibration record
//! established at one target. This module declares those objects as a [`TransportScene`],
//! builds the claim-level dependency graph of [`crate::composition_verification`] from
//! it, and classifies every transported claim and every decision resting on one by
//! comparing two scenes with the existing [`revalidate`]. The graph and its
//! classification rules are not changed here; this module only fixes how transport
//! objects map onto nodes, edges and coordinates, and names the reasons in transport
//! terms.
//!
//! The mapping:
//!
//! * the target population is a node; every transported claim depends on it by a support
//!   dependency, and a claim's evaluation coordinate is the pair (target, quantity);
//! * the selection diagram is a node; every transported claim reuses its structural
//!   derivation, so a changed diagram recomputes the claim and retains no earlier fit;
//! * every source is a node; a claim counts the evidence of the sources it names;
//! * a decision depends on the claims it rests on.
//!
//! Classification:
//!
//! * `recompute`: the target population, a source's evidence, the selection diagram or
//!   the claim's own derivation changed, or an observed effect reports it stale;
//! * `withdrawn`: the target or the claim is outside its support, a provider's trust no
//!   longer holds, an input was retired, or the claim's calibration does not hold at its
//!   coordinate;
//! * `still_valid`: nothing it rests on changed, and the earlier numbers stand.
//!
//! Transported calibration is established at one target. A record held at one target is
//! never read as holding at another: a claim whose record was established at a different
//! target than the one it now answers is withdrawn for calibration with the typed reason
//! [`TransportReason::CalibrationFromAnotherTarget`], and its point value is left as it
//! was. A claim that inherits a record from another claim holds it only while that claim
//! holds it at the same target and quantity.
//!
//! # Refresh
//!
//! New outcome-only data at a source population changes that source's identity: the
//! claims that count it are *recomputed*; the scene's declaration (diagram, quantity
//! coordinates, claim wiring) is *kept*; a claim's calibration record is never carried
//! to another target (it is withdrawn with
//! [`TransportReason::CalibrationFromAnotherTarget`]) and data of another population,
//! quantity or arm is a different scene, not a refresh. Revalidation never edits a scene:
//! it compares two and reports, so a refused or withdrawn claim leaves the earlier
//! numbers untouched.
//!
//! This module computes no statistic and declares no calibration record.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use antecedent_core::{CanonicalWriter, SemanticDigest};

use crate::ExternalRefusal;
use crate::composition_verification::{
    CalibrationDeclaration, CalibrationState, ClaimRevalidation, CompositionGraph, CompositionNode,
    DependencyKind, GraphError, GraphRevalidation, NodeKind, NumericalCause, ObservedEffect,
    PriorAuthority, Revalidation, SupportStanding, TrustStanding, WithdrawalKind, revalidate,
};

/// Version word written into the identity of a transport revalidation.
pub const TRANSPORT_REVALIDATION_VERSION: u64 = 1;

const TARGET_KEY: &str = "target_population";
const DIAGRAM_KEY: &str = "selection_diagram";
const SOURCE_PREFIX: &str = "source:";
const CLAIM_PREFIX: &str = "claim:";

/// The target population a scene answers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SceneTarget {
    /// Content identity of the target population's declaration and data.
    pub identity: SemanticDigest,
    /// Whether the target lies inside the support the transport needs (positivity,
    /// overlap, a mapped selection variable).
    pub support: SupportStanding,
}

/// One source of evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SceneSource {
    /// Stable key, unique within the scene.
    pub key: String,
    /// Content identity of the source's data and declaration.
    pub identity: SemanticDigest,
}

/// How a transported claim's uncertainty guarantee is backed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SceneCalibration {
    /// A point value; no guarantee is claimed.
    PointOnly,
    /// A guarantee is wanted and no record backs it.
    Required,
    /// A record the claim holds itself, established at `established_at_target`.
    Held {
        /// Identity of the calibration record.
        record: SemanticDigest,
        /// Identity of the target population the record was established at.
        established_at_target: SemanticDigest,
    },
    /// A record claimed from another claim in the scene.
    Inherited {
        /// Key of the claim the record is claimed from.
        from: String,
        /// Identity of the calibration record.
        record: SemanticDigest,
        /// Identity of the target population the record was established at.
        established_at_target: SemanticDigest,
    },
}

/// What a claim in the scene is.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum SceneClaimKind {
    /// A transported claim about the target.
    Transported,
    /// A decision resting on claims.
    Decision,
}

/// One transported claim or a decision resting on claims.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SceneClaim {
    /// Stable key, unique among claims.
    pub key: String,
    /// What it is.
    pub kind: SceneClaimKind,
    /// Content identity of its numbers and declarations.
    pub identity: SemanticDigest,
    /// Identity of its structural derivation (identification, bridge form), when it
    /// has one.
    pub structure: Option<SemanticDigest>,
    /// Identity of the quantity coordinate it answers (units, regime, horizon,
    /// functional, conditioning).
    pub quantity: SemanticDigest,
    /// Keys of the sources whose evidence it counts.
    pub sources: Vec<String>,
    /// Keys of the claims it rests on.
    pub rests_on: Vec<String>,
    /// Its own coordinates outside the support, if any.
    pub support: SupportStanding,
    /// Provider trust, when it is a provider's output.
    pub trust: TrustStanding,
    /// Its uncertainty guarantee.
    pub calibration: SceneCalibration,
}

impl SceneClaim {
    /// A supported point-only transported claim with no structure, sources or
    /// dependencies.
    #[must_use]
    pub fn new(
        key: impl Into<String>,
        kind: SceneClaimKind,
        identity: SemanticDigest,
        quantity: SemanticDigest,
    ) -> Self {
        Self {
            key: key.into(),
            kind,
            identity,
            structure: None,
            quantity,
            sources: Vec::new(),
            rests_on: Vec::new(),
            support: SupportStanding::Supported,
            trust: TrustStanding::NotApplicable,
            calibration: SceneCalibration::PointOnly,
        }
    }
}

/// Everything that decides which claims about one target are licensed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransportScene {
    /// The target population.
    pub target: SceneTarget,
    /// Content identity of the selection diagram.
    pub selection_diagram: SemanticDigest,
    /// The sources.
    pub sources: Vec<SceneSource>,
    /// The claims and decisions.
    pub claims: Vec<SceneClaim>,
}

/// The coordinate a claim at `target` answering `quantity` is evaluated at; also the
/// coordinate a calibration record established at `target` for `quantity` holds at.
#[must_use]
pub fn transport_coordinate(target: SemanticDigest, quantity: SemanticDigest) -> SemanticDigest {
    let mut w = CanonicalWriter::new("antecedent.transport_revalidation.coordinate");
    w.bytes(target.as_bytes()).bytes(quantity.as_bytes());
    SemanticDigest::from_bytes(w.finish())
}

fn claim_key(key: &str) -> String {
    format!("{CLAIM_PREFIX}{key}")
}

fn source_key(key: &str) -> String {
    format!("{SOURCE_PREFIX}{key}")
}

impl TransportScene {
    /// The claim-level graph of the scene.
    ///
    /// # Errors
    /// A duplicate key, a reference to a source or claim the scene does not declare, a
    /// dependency cycle, a claim inheriting from a claim it does not rest on or from
    /// an unknown claim, or a bound exceeded.
    pub fn graph(&self) -> Result<CompositionGraph, GraphError> {
        let mut graph = CompositionGraph::new();
        let mut target =
            CompositionNode::new(TARGET_KEY, NodeKind::TargetPopulation, self.target.identity)
                .with_coordinate(self.target.identity);
        target.support = self.target.support.clone();
        graph.add_node(target)?;
        graph.add_node(CompositionNode::new(
            DIAGRAM_KEY,
            NodeKind::SelectionDiagram,
            self.selection_diagram,
        ))?;
        for source in &self.sources {
            graph.add_node(CompositionNode::new(
                source_key(&source.key),
                NodeKind::CohortSource,
                source.identity,
            ))?;
        }
        let quantities: BTreeMap<&str, SemanticDigest> =
            self.claims.iter().map(|c| (c.key.as_str(), c.quantity)).collect();
        for claim in &self.claims {
            let coordinate = transport_coordinate(self.target.identity, claim.quantity);
            let kind = match claim.kind {
                SceneClaimKind::Transported => NodeKind::TransportedResult,
                SceneClaimKind::Decision => NodeKind::Decision,
            };
            let calibration = match &claim.calibration {
                SceneCalibration::PointOnly => CalibrationDeclaration::PointOnly,
                SceneCalibration::Required => CalibrationDeclaration::Required,
                SceneCalibration::Held { record, established_at_target } => {
                    CalibrationDeclaration::Held {
                        record: *record,
                        coordinate: transport_coordinate(*established_at_target, claim.quantity),
                    }
                }
                SceneCalibration::Inherited { from, record, established_at_target } => {
                    let Some(up_quantity) = quantities.get(from.as_str()) else {
                        return Err(GraphError::UnknownNode(claim_key(from)));
                    };
                    CalibrationDeclaration::Inherited {
                        from: claim_key(from),
                        record: *record,
                        coordinate: transport_coordinate(*established_at_target, *up_quantity),
                    }
                }
            };
            let mut node = CompositionNode::new(claim_key(&claim.key), kind, claim.identity)
                .with_coordinate(coordinate)
                .with_trust(claim.trust)
                .with_calibration(calibration);
            node.structure = claim.structure;
            node.support = claim.support.clone();
            graph.add_node(node)?;
        }
        for claim in &self.claims {
            let key = claim_key(&claim.key);
            graph.add_edge(&key, TARGET_KEY, DependencyKind::Support)?;
            if claim.kind == SceneClaimKind::Transported {
                graph.add_edge(&key, DIAGRAM_KEY, DependencyKind::StructuralDerivation)?;
            }
            for source in &claim.sources {
                graph.add_edge(&key, &source_key(source), DependencyKind::Evidence)?;
            }
            for upstream in &claim.rests_on {
                graph.add_edge(&key, &claim_key(upstream), DependencyKind::Data)?;
            }
            if let SceneCalibration::Inherited { from, .. } = &claim.calibration {
                graph.add_edge(&key, &claim_key(from), DependencyKind::Calibration)?;
            }
        }
        Ok(graph)
    }
}

/// Why a transported claim is not still valid, in transport terms.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum TransportReason {
    /// The target population changed: the claim answers a different population.
    TargetPopulationChanged,
    /// A source's evidence changed.
    SourceEvidenceChanged {
        /// The source key.
        source: String,
    },
    /// The selection diagram changed.
    SelectionDiagramChanged,
    /// The claim's own derivation, quantity or dependency set changed.
    ClaimChanged {
        /// The claim whose own input changed.
        claim: String,
        /// What changed there.
        cause: NumericalCause,
    },
    /// The claim is new in this scene.
    ClaimAdded {
        /// The claim.
        claim: String,
    },
    /// An observed effect (a stale stage or ledger result) reports a node stale.
    ObservedStale {
        /// The node key.
        node: String,
    },
    /// The target lies outside the support the transport needs.
    TargetUnsupported {
        /// Named coordinates outside the support.
        coordinates: String,
    },
    /// The claim's own coordinates lie outside the support.
    ClaimUnsupported {
        /// The claim.
        claim: String,
        /// Named coordinates outside the support.
        coordinates: String,
    },
    /// A provider the claim relies on no longer holds its trust.
    ProviderTrustLost {
        /// The claim carrying the provider's output.
        claim: String,
    },
    /// An input the claim cannot do without was retired, replaced, refused or
    /// removed.
    InputRetired {
        /// The input key.
        input: String,
    },
    /// The claim's calibration record was established at a different target (or for a
    /// different quantity) than the one the claim now answers.
    CalibrationFromAnotherTarget,
    /// The claim inherits a calibration the upstream claim no longer holds.
    CalibrationUpstreamNotHeld,
    /// The claim inherits a record that differs from the one the upstream claim holds.
    CalibrationRecordDiffers,
}

impl TransportReason {
    /// Whether the reason withdraws a license (as opposed to staling the numbers).
    #[must_use]
    pub const fn is_withdrawal(&self) -> bool {
        matches!(
            self,
            Self::TargetUnsupported { .. }
                | Self::ClaimUnsupported { .. }
                | Self::ProviderTrustLost { .. }
                | Self::InputRetired { .. }
                | Self::CalibrationFromAnotherTarget
                | Self::CalibrationUpstreamNotHeld
                | Self::CalibrationRecordDiffers
        )
    }

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::TargetPopulationChanged => "target_population_changed",
            Self::SourceEvidenceChanged { .. } => "source_evidence_changed",
            Self::SelectionDiagramChanged => "selection_diagram_changed",
            Self::ClaimChanged { .. } => "claim_changed",
            Self::ClaimAdded { .. } => "claim_added",
            Self::ObservedStale { .. } => "observed_stale",
            Self::TargetUnsupported { .. } => "target_unsupported",
            Self::ClaimUnsupported { .. } => "claim_unsupported",
            Self::ProviderTrustLost { .. } => "provider_trust_lost",
            Self::InputRetired { .. } => "input_retired",
            Self::CalibrationFromAnotherTarget => "calibration_from_another_target",
            Self::CalibrationUpstreamNotHeld => "calibration_upstream_not_held",
            Self::CalibrationRecordDiffers => "calibration_record_differs",
        }
    }

    /// What the caller can do about it.
    #[must_use]
    pub const fn remedy(&self) -> &'static str {
        match self {
            Self::TargetPopulationChanged
            | Self::SourceEvidenceChanged { .. }
            | Self::SelectionDiagramChanged
            | Self::ClaimChanged { .. }
            | Self::ClaimAdded { .. }
            | Self::ObservedStale { .. } => {
                "recompute the transported claim from the changed inputs; the earlier fit and \
                 uncertainty carry no authority"
            }
            Self::TargetUnsupported { .. } | Self::ClaimUnsupported { .. } => {
                "restrict the claim to coordinates inside the support, or supply target \
                 evidence that extends it"
            }
            Self::ProviderTrustLost { .. } => {
                "re-verify the provider at its current identity, or use a native route"
            }
            Self::InputRetired { .. } => {
                "recompute the claim from the successor of the retired input, or declare a \
                 replacement"
            }
            Self::CalibrationFromAnotherTarget
            | Self::CalibrationUpstreamNotHeld
            | Self::CalibrationRecordDiffers => {
                "establish a calibration record at this target and quantity, or report the \
                 point value without an interval or posterior"
            }
        }
    }
}

/// One claim's classification in transport terms.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransportClaimRevalidation {
    /// The claim key as declared in the scene.
    pub key: String,
    /// What the claim is.
    pub kind: SceneClaimKind,
    /// `still_valid`, `recompute` or `withdrawn`.
    pub class: Revalidation,
    /// Why, sorted and distinct. Empty only for a `still_valid` claim.
    pub reasons: Vec<TransportReason>,
    /// Whether the earlier numbers and uncertainty may still be used.
    pub authority: PriorAuthority,
    /// The claim's uncertainty guarantee after revalidation.
    pub calibration: CalibrationState,
    inner: ClaimRevalidation,
}

impl TransportClaimRevalidation {
    /// The classification of the underlying graph node.
    #[must_use]
    pub const fn graph_claim(&self) -> &ClaimRevalidation {
        &self.inner
    }

    /// The typed refusal for a claim that is not usable as it stands, or `None` for a
    /// `still_valid` claim.
    #[must_use]
    pub fn refusal(&self) -> Option<ExternalRefusal> {
        self.inner.refusal()
    }

    /// Whether the claim's calibration was withdrawn.
    #[must_use]
    pub fn calibration_withdrawn(&self) -> bool {
        self.inner.withdrawn_for(WithdrawalKind::Calibration)
    }
}

/// Classification of every claim of a scene against an earlier scene.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransportRevalidation {
    claims: BTreeMap<String, TransportClaimRevalidation>,
    removed: Vec<String>,
    graph: GraphRevalidation,
}

impl TransportRevalidation {
    /// The classification of the claim `key`.
    #[must_use]
    pub fn claim(&self, key: &str) -> Option<&TransportClaimRevalidation> {
        self.claims.get(key)
    }

    /// Every claim in key order.
    pub fn claims(&self) -> impl Iterator<Item = &TransportClaimRevalidation> {
        self.claims.values()
    }

    /// Claim keys of class `class`, sorted.
    #[must_use]
    pub fn keys_of(&self, class: Revalidation) -> Vec<&str> {
        self.claims.values().filter(|c| c.class == class).map(|c| c.key.as_str()).collect()
    }

    /// Claims present in the earlier scene and absent now, sorted.
    #[must_use]
    pub fn removed(&self) -> &[String] {
        &self.removed
    }

    /// The underlying graph classification.
    #[must_use]
    pub const fn graph(&self) -> &GraphRevalidation {
        &self.graph
    }

    /// Identity of the classification: the graph classification and every claim's
    /// transport reasons, length-delimited.
    #[must_use]
    pub fn identity(&self) -> SemanticDigest {
        let mut w = CanonicalWriter::new("antecedent.transport_revalidation.result");
        w.word(TRANSPORT_REVALIDATION_VERSION).bytes(self.graph.identity().as_bytes());
        w.word(self.claims.len() as u64);
        for claim in self.claims.values() {
            w.text(&claim.key).text(claim.class.as_str()).word(claim.reasons.len() as u64);
            for reason in &claim.reasons {
                w.text(reason.as_str());
            }
        }
        SemanticDigest::from_bytes(w.finish())
    }
}

fn strip<'a>(key: &'a str, prefix: &str) -> Option<&'a str> {
    key.strip_prefix(prefix)
}

fn numerical_reason(origin: &str, cause: NumericalCause) -> TransportReason {
    if origin == TARGET_KEY {
        TransportReason::TargetPopulationChanged
    } else if origin == DIAGRAM_KEY {
        TransportReason::SelectionDiagramChanged
    } else if let Some(source) = strip(origin, SOURCE_PREFIX) {
        TransportReason::SourceEvidenceChanged { source: source.to_owned() }
    } else {
        let claim = strip(origin, CLAIM_PREFIX).unwrap_or(origin).to_owned();
        match cause {
            NumericalCause::Added => TransportReason::ClaimAdded { claim },
            NumericalCause::Observed => TransportReason::ObservedStale { node: claim },
            cause => TransportReason::ClaimChanged { claim, cause },
        }
    }
}

fn withdrawal_reason(kind: WithdrawalKind, origin: &str, detail: &str) -> TransportReason {
    let name = strip(origin, CLAIM_PREFIX)
        .or_else(|| strip(origin, SOURCE_PREFIX))
        .unwrap_or(origin)
        .to_owned();
    match kind {
        WithdrawalKind::Support if origin == TARGET_KEY => {
            TransportReason::TargetUnsupported { coordinates: detail.to_owned() }
        }
        WithdrawalKind::Support => {
            TransportReason::ClaimUnsupported { claim: name, coordinates: detail.to_owned() }
        }
        WithdrawalKind::ProviderTrust => TransportReason::ProviderTrustLost { claim: name },
        WithdrawalKind::Retired | WithdrawalKind::Unexecutable => {
            TransportReason::InputRetired { input: name }
        }
        WithdrawalKind::Calibration => match detail {
            "upstream_calibration_not_held" => TransportReason::CalibrationUpstreamNotHeld,
            "record_differs" => TransportReason::CalibrationRecordDiffers,
            _ => TransportReason::CalibrationFromAnotherTarget,
        },
    }
}

/// Classify every claim and decision of `current` against `previous`.
///
/// `effects` carries what a recalculation plan, the evidence ledger or claim freshness
/// say about individual nodes, keyed by the node keys of [`TransportScene::graph`]
/// (`claim:<key>`, `source:<key>`, `target_population`, `selection_diagram`).
///
/// # Errors
/// A scene whose graph does not build or validate.
pub fn revalidate_transport(
    previous: &TransportScene,
    current: &TransportScene,
    effects: &BTreeMap<String, ObservedEffect>,
) -> Result<TransportRevalidation, GraphError> {
    let before = previous.graph()?;
    let after = current.graph()?;
    let graph = revalidate(&before, &after, effects)?;
    let mut claims = BTreeMap::new();
    for claim in &current.claims {
        let node_key = claim_key(&claim.key);
        let inner = graph
            .claim(&node_key)
            .cloned()
            .ok_or_else(|| GraphError::UnknownNode(node_key.clone()))?;
        let mut reasons: BTreeSet<TransportReason> = BTreeSet::new();
        for numerical in &inner.numerical {
            reasons.insert(numerical_reason(&numerical.origin, numerical.cause));
        }
        for withdrawal in &inner.withdrawals {
            reasons.insert(withdrawal_reason(
                withdrawal.kind,
                &withdrawal.origin,
                &withdrawal.detail,
            ));
        }
        claims.insert(
            claim.key.clone(),
            TransportClaimRevalidation {
                key: claim.key.clone(),
                kind: claim.kind,
                class: inner.class,
                reasons: reasons.into_iter().collect(),
                authority: inner.authority,
                calibration: inner.calibration.state,
                inner,
            },
        );
    }
    let current_keys: BTreeSet<&str> = current.claims.iter().map(|c| c.key.as_str()).collect();
    let removed = previous
        .claims
        .iter()
        .filter(|c| !current_keys.contains(c.key.as_str()))
        .map(|c| c.key.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    Ok(TransportRevalidation { claims, removed, graph })
}
