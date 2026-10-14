//! Classification of every claim in a graph against an earlier snapshot.
//!
//! The rules, in the order they apply to one node (nodes are visited with
//! every dependency first):
//!
//! 1. **Own change.** A node that is new, whose content identity, structural
//!    identity, representation version, coordinate or dependency set differs
//!    from the earlier snapshot, or that an observed effect (a recomputed
//!    stage, a stale ledger result) reports stale, has a *numerical reason*.
//! 2. **Own withdrawal.** A node whose support does not cover its coordinates,
//!    whose provider trust no longer holds, or whose observed effect is a
//!    withdrawal or a refusal, has a *withdrawal*.
//! 3. **Upstream.** Along each dependency, upstream numerical reasons reach the
//!    dependent (data replaced upstream means numbers recomputed here), and
//!    upstream withdrawals reach the dependent: a retired, unsupported,
//!    untrusted or unexecutable input is one the claim cannot do without. A
//!    calibration withdrawal reaches the dependent only along a calibration
//!    dependency.
//!    A change of standing between the snapshots is itself an own change: a
//!    provider trust standing that moved without being withdrawn, a support
//!    that was restored, or a calibration record replaced by another one means
//!    the earlier numbers or uncertainty carry no authority even when every
//!    stage that produced them was reused.
//! 4. **Calibration.** A claim that holds a record must be at the record's
//!    coordinate. A claim that inherits one must have an upstream that
//!    currently holds the same record at the same coordinate. Otherwise the
//!    claim is withdrawn for calibration; the point value is not touched.
//!
//! The class is `withdrawn` when any withdrawal stands, `recompute` when only
//! numerical reasons stand, and `still_valid` otherwise. Numerical
//! invalidation and withdrawal are kept apart in the result: a claim can need
//! recomputation without losing any license, and can be withdrawn with its
//! numbers intact. The earlier fit and uncertainty are retained only by a
//! `still_valid` claim: reusing a structural derivation after the data were
//! replaced recomputes the numbers and keeps no authority for the old fit.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use antecedent_core::recalc::StageStatus;
use antecedent_core::{CanonicalWriter, ContractIdentities, SemanticDigest};

use crate::ExternalRefusal;
use crate::claim::{ClaimEnvelope, ClaimFreshness};
use crate::evidence_ledger::{EvidenceLedger, ResultId, ResultStatus, Revalidation};

use super::graph::{
    CalibrationDeclaration, CompositionGraph, DependencyKind, GraphError, NodeKind,
    SupportStanding, TrustStanding,
};

/// What an observation outside the graph says about one node.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ObservedEffect {
    /// The node's numbers are stale: recompute them.
    Recomputed {
        /// Where the observation came from (a stage status, an operation).
        cause: String,
    },
    /// The node was withdrawn, replaced or retired.
    Withdrawn {
        /// Where the observation came from.
        cause: String,
    },
    /// The node cannot be served (a refused stage).
    Refused {
        /// The refusal detail.
        detail: String,
    },
}

impl ObservedEffect {
    /// The effect of one stage's status in a recalculation plan: reuse is no
    /// effect, recomputation is a numerical effect, a refused stage cannot be
    /// executed.
    #[must_use]
    pub fn from_stage_status(status: &StageStatus) -> Option<Self> {
        match status {
            StageStatus::Reused { .. } => None,
            StageStatus::Recomputed { because } => {
                Some(Self::Recomputed { cause: format!("stage_status:{because}") })
            }
            StageStatus::Refused { reason } => Some(Self::Refused { detail: reason.to_string() }),
        }
    }

    /// The effect of a ledger result's standing.
    #[must_use]
    pub fn from_result_status(status: &ResultStatus) -> Option<Self> {
        match status {
            ResultStatus::Valid => None,
            ResultStatus::NeedsRecompute { cause } => {
                Some(Self::Recomputed { cause: format!("ledger.operation:{cause}") })
            }
            ResultStatus::Withdrawn { cause } => {
                Some(Self::Withdrawn { cause: format!("ledger.operation:{cause}") })
            }
            ResultStatus::Replaced { by, .. } => {
                Some(Self::Withdrawn { cause: format!("ledger.replaced_by:{by}") })
            }
        }
    }

    /// The effect of a claim's freshness: withdrawal is a withdrawal, a
    /// recorded successor and a snapshot that no longer matches both ask for
    /// recomputation. An offline receiver with no current snapshot learns
    /// nothing and reports no effect.
    #[must_use]
    pub fn from_freshness(freshness: &ClaimFreshness) -> Option<Self> {
        if freshness.withdrawn {
            return Some(Self::Withdrawn { cause: "claim.withdrawn".to_owned() });
        }
        if freshness.superseded_by.is_some() {
            return Some(Self::Recomputed { cause: "claim.superseded".to_owned() });
        }
        (freshness.currently_applicable == Some(false))
            .then(|| Self::Recomputed { cause: "claim.snapshot_changed".to_owned() })
    }
}

impl ObservedEffect {
    /// The effect of a claim against the contract identities the receiver now
    /// holds: withdrawal and supersession as in [`Self::from_freshness`], and
    /// otherwise every identity layer (target, identification and its product,
    /// program, inference binding, observation, data snapshot) is compared, so a
    /// change in any layer with an unchanged data snapshot is not read as
    /// applicable. The changed layers are named in the cause. With no current
    /// identities the receiver learns nothing and the freshness reading stands.
    #[must_use]
    pub fn from_claim_contract(
        claim: &ClaimEnvelope,
        freshness: &ClaimFreshness,
        current: Option<&ContractIdentities>,
    ) -> Option<Self> {
        if freshness.withdrawn {
            return Some(Self::Withdrawn { cause: "claim.withdrawn".to_owned() });
        }
        if freshness.superseded_by.is_some() {
            return Some(Self::Recomputed { cause: "claim.superseded".to_owned() });
        }
        let Some(now) = current else {
            return Self::from_freshness(freshness);
        };
        let was = &claim.identities;
        let mut changed: Vec<&'static str> = Vec::new();
        if was.target != now.target {
            changed.push("target");
        }
        if was.identification != now.identification {
            changed.push("identification");
        }
        if was.identification_product != now.identification_product {
            changed.push("identification_product");
        }
        if was.program != now.program {
            changed.push("program");
        }
        if was.inference_binding != now.inference_binding {
            changed.push("inference_binding");
        }
        if was.observation != now.observation {
            changed.push("observation");
        }
        if was.data_snapshot != now.data_snapshot {
            changed.push("data_snapshot");
        }
        if changed.is_empty() {
            return None;
        }
        Some(Self::Recomputed { cause: format!("claim.contract_changed:{}", changed.join(",")) })
    }

    /// The effect of a class decided by another vocabulary that adapts onto
    /// the one three-state classification (the state's invalidation log, a
    /// discovery or counterfactual stage receipt, a callback replay plan):
    /// `still_valid` is no effect, `recompute` a numerical effect, `withdrawn`
    /// a withdrawal. `cause` names where the class came from.
    #[must_use]
    pub fn from_revalidation(class: Revalidation, cause: impl Into<String>) -> Option<Self> {
        match class {
            Revalidation::StillValid => None,
            Revalidation::Recompute => Some(Self::Recomputed { cause: cause.into() }),
            Revalidation::Withdrawn => Some(Self::Withdrawn { cause: cause.into() }),
        }
    }
}

/// The effects of every bound ledger result, keyed by node key.
///
/// # Errors
/// A result id that is not in the ledger.
pub fn ledger_effects(
    ledger: &EvidenceLedger,
    bindings: &[(&str, ResultId)],
) -> Result<BTreeMap<String, ObservedEffect>, GraphError> {
    let mut out = BTreeMap::new();
    for (key, id) in bindings {
        let Some(record) = ledger.result(*id) else {
            return Err(GraphError::UnknownNode(format!("ledger result {id}")));
        };
        if let Some(effect) = ObservedEffect::from_result_status(&record.status) {
            out.insert((*key).to_owned(), effect);
        }
    }
    Ok(out)
}

/// Why a node's numbers must be recomputed.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum NumericalCause {
    /// The node has no earlier result to reuse.
    Added,
    /// The content identity changed (data, fit or declarations replaced).
    IdentityChanged,
    /// The structural derivation itself changed.
    StructureChanged,
    /// The structural derivation is the same but the fit it was reused with
    /// was replaced: the structure is reusable, the old fit and uncertainty
    /// authority are not.
    StructureReusedFitNotRetained,
    /// The model or representation version changed.
    RepresentationChanged,
    /// The evaluation coordinate changed.
    CoordinateChanged,
    /// The set of dependencies changed.
    DependenciesChanged,
    /// An observed effect reports the node stale.
    Observed,
    /// The provider trust standing moved (and does not withdraw): the numbers
    /// were produced under a different standing.
    TrustChanged,
    /// The node's calibration record was replaced by a different one: the
    /// earlier uncertainty statement carries no authority.
    CalibrationChanged,
    /// Coordinates that were outside the support are supported again: the
    /// earlier numbers were never licensed there.
    SupportRestored,
}

impl NumericalCause {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Added => "added",
            Self::IdentityChanged => "identity_changed",
            Self::StructureChanged => "structure_changed",
            Self::StructureReusedFitNotRetained => "structure_reused_fit_not_retained",
            Self::RepresentationChanged => "representation_changed",
            Self::CoordinateChanged => "coordinate_changed",
            Self::DependenciesChanged => "dependencies_changed",
            Self::Observed => "observed",
            Self::TrustChanged => "trust_changed",
            Self::CalibrationChanged => "calibration_changed",
            Self::SupportRestored => "support_restored",
        }
    }
}

/// One reason a node's numbers are stale.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NumericalReason {
    /// The node whose own input changed.
    pub origin: String,
    /// The dependency the change arrived through, and how it depends.
    pub via: Option<(String, DependencyKind)>,
    /// What changed at the origin.
    pub cause: NumericalCause,
}

/// What kind of license was lost.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum WithdrawalKind {
    /// An input it cannot do without was retired, replaced or removed.
    Retired,
    /// Coordinates it answers are outside the support.
    Support,
    /// A provider it relies on no longer holds its trust.
    ProviderTrust,
    /// The calibration it states does not hold at its coordinate.
    Calibration,
    /// A stage it needs cannot be executed.
    Unexecutable,
}

impl WithdrawalKind {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Retired => "retired",
            Self::Support => "support",
            Self::ProviderTrust => "provider_trust",
            Self::Calibration => "calibration",
            Self::Unexecutable => "unexecutable",
        }
    }
}

/// One withdrawn license.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Withdrawal {
    /// What was lost.
    pub kind: WithdrawalKind,
    /// The node the loss originates at.
    pub origin: String,
    /// The dependency it arrived through, when not the node's own.
    pub via: Option<(String, DependencyKind)>,
    /// A stable slot or the coordinates concerned.
    pub detail: String,
}

/// Whether the earlier fit and uncertainty may still be used.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PriorAuthority {
    /// Nothing it rests on changed: the earlier numbers stand.
    Retained,
    /// The earlier fit and its uncertainty carry no authority.
    NotRetained,
}

/// The state of a node's uncertainty guarantee.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum CalibrationState {
    /// Point only: no guarantee is claimed.
    NotApplicable,
    /// A record held at the node's own coordinate.
    Held,
    /// A record inherited from an upstream node that currently holds it.
    Inherited,
    /// A record the node states but that does not hold.
    Lost,
    /// A guarantee is wanted and no record backs it.
    Required,
}

impl CalibrationState {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NotApplicable => "not_applicable",
            Self::Held => "held",
            Self::Inherited => "inherited",
            Self::Lost => "lost",
            Self::Required => "required",
        }
    }
}

/// Calibration inherited, lost or newly required for one claim.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CalibrationLineage {
    /// The state.
    pub state: CalibrationState,
    /// The upstream node the record was claimed from.
    pub from: Option<String>,
    /// Whether a new calibration must be established at this coordinate
    /// before an uncertainty statement may be made.
    pub newly_required: bool,
    /// Why a record does not hold.
    pub detail: Option<&'static str>,
}

impl CalibrationLineage {
    const fn plain(state: CalibrationState) -> Self {
        Self { state, from: None, newly_required: false, detail: None }
    }
}

/// The classification of one claim.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClaimRevalidation {
    /// The node key.
    pub key: String,
    /// What the node is.
    pub kind: NodeKind,
    /// The classification.
    pub class: Revalidation,
    /// Why the numbers are stale (sorted, distinct).
    pub numerical: Vec<NumericalReason>,
    /// Which licenses were lost (sorted, distinct).
    pub withdrawals: Vec<Withdrawal>,
    /// Whether the earlier fit and uncertainty stand.
    pub authority: PriorAuthority,
    /// The node's uncertainty guarantee.
    pub calibration: CalibrationLineage,
}

impl ClaimRevalidation {
    /// Whether a withdrawal of the given kind stands.
    #[must_use]
    pub fn withdrawn_for(&self, kind: WithdrawalKind) -> bool {
        self.withdrawals.iter().any(|w| w.kind == kind)
    }

    /// The typed refusal for a claim that is not usable as it stands, or
    /// `None` for a `still_valid` claim.
    #[must_use]
    pub fn refusal(&self) -> Option<ExternalRefusal> {
        let stage = "revalidate";
        if let Some(w) = self.withdrawals.iter().min_by_key(|w| (w.kind, &w.origin)) {
            let (code, detail, remedy) = match w.kind {
                WithdrawalKind::Calibration => (
                    antecedent_core::reason_code!("cell_not_licensed"),
                    "composition_verification.calibration_not_held",
                    "establish a calibration record at this claim's coordinate, or report the \
                     point value without an interval or posterior",
                ),
                WithdrawalKind::Support => (
                    antecedent_core::reason_code!("claim_license_withdrawn"),
                    "composition_verification.support_withdrawn",
                    "restrict the claim to coordinates inside the support, or supply evidence \
                     that extends it",
                ),
                WithdrawalKind::ProviderTrust => (
                    antecedent_core::reason_code!("claim_license_withdrawn"),
                    "composition_verification.provider_trust_withdrawn",
                    "re-verify the provider at its current identity, or use a native route",
                ),
                WithdrawalKind::Retired => (
                    antecedent_core::reason_code!("claim_license_withdrawn"),
                    "composition_verification.input_retired",
                    "recompute the claim from the successor of the retired input, or declare \
                     a replacement",
                ),
                WithdrawalKind::Unexecutable => (
                    antecedent_core::reason_code!("claim_license_withdrawn"),
                    "composition_verification.stage_unexecutable",
                    "supply the data, fit or provider the stage needs, or recompute it",
                ),
            };
            return Some(
                ExternalRefusal::new(code, stage, detail)
                    .with_offending(self.key.clone())
                    .with_expected(format!("origin {}", w.origin))
                    .with_supplied(w.kind.as_str())
                    .with_remedy(remedy),
            );
        }
        if self.class == Revalidation::Recompute {
            let origin = self.numerical.first().map_or("unknown", |r| r.origin.as_str());
            return Some(
                ExternalRefusal::new(
                    antecedent_core::reason_code!("claim_revalidation_required"),
                    stage,
                    "composition_verification.recompute_required",
                )
                .with_offending(self.key.clone())
                .with_expected(format!("origin {origin}"))
                .with_remedy(
                    "recompute the claim from the changed upstream objects; the earlier fit and \
                     uncertainty carry no authority",
                ),
            );
        }
        None
    }
}

/// Classification of a whole graph.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphRevalidation {
    claims: BTreeMap<String, ClaimRevalidation>,
    removed: Vec<String>,
    previous: SemanticDigest,
    current: SemanticDigest,
}

impl GraphRevalidation {
    /// The classification of `key`.
    #[must_use]
    pub fn claim(&self, key: &str) -> Option<&ClaimRevalidation> {
        self.claims.get(key)
    }

    /// Every classification in key order.
    pub fn claims(&self) -> impl Iterator<Item = &ClaimRevalidation> {
        self.claims.values()
    }

    /// Keys with class `class`, sorted.
    #[must_use]
    pub fn keys_of(&self, class: Revalidation) -> Vec<&str> {
        self.claims.values().filter(|c| c.class == class).map(|c| c.key.as_str()).collect()
    }

    /// Nodes present earlier and absent now, sorted.
    #[must_use]
    pub fn removed(&self) -> &[String] {
        &self.removed
    }

    /// Identity of the earlier snapshot.
    #[must_use]
    pub const fn previous_identity(&self) -> SemanticDigest {
        self.previous
    }

    /// Identity of the current snapshot.
    #[must_use]
    pub const fn current_identity(&self) -> SemanticDigest {
        self.current
    }

    /// The claim, or its typed refusal when it is not usable as it stands.
    ///
    /// # Errors
    /// The refusal for a `recompute` or `withdrawn` claim, or an unknown key.
    #[allow(
        clippy::result_large_err,
        reason = "a refusal is the cold path of a once-per-claim check; boxing would not pay"
    )]
    pub fn require_valid(&self, key: &str) -> Result<&ClaimRevalidation, ExternalRefusal> {
        let Some(claim) = self.claims.get(key) else {
            return Err(ExternalRefusal::new(
                antecedent_core::reason_code!("invalid_argument"),
                "revalidate",
                "composition_verification.unknown_claim",
            )
            .with_offending(key.to_owned())
            .with_remedy("name a claim declared in the current composition graph"));
        };
        claim.refusal().map_or(Ok(claim), Err)
    }

    /// Identity of the classification: graph identities and every claim's
    /// class, reasons, withdrawals, authority and calibration.
    #[must_use]
    pub fn identity(&self) -> SemanticDigest {
        let mut w = CanonicalWriter::new("antecedent.composition_verification.revalidation");
        w.bytes(self.previous.as_bytes()).bytes(self.current.as_bytes());
        w.texts(&self.removed).word(self.claims.len() as u64);
        for claim in self.claims.values() {
            w.text(&claim.key).text(claim.class.as_str());
            w.word(claim.numerical.len() as u64);
            for r in &claim.numerical {
                w.text(&r.origin).text(r.cause.as_str());
                match &r.via {
                    Some((up, kind)) => w.word(1).text(up).text(kind.as_str()),
                    None => w.word(0),
                };
            }
            w.word(claim.withdrawals.len() as u64);
            for wd in &claim.withdrawals {
                w.text(wd.kind.as_str()).text(&wd.origin).text(&wd.detail);
                match &wd.via {
                    Some((up, kind)) => w.word(1).text(up).text(kind.as_str()),
                    None => w.word(0),
                };
            }
            w.word(u64::from(claim.authority == PriorAuthority::Retained));
            w.text(claim.calibration.state.as_str())
                .opt_text(claim.calibration.from.as_deref())
                .word(u64::from(claim.calibration.newly_required))
                .opt_text(claim.calibration.detail);
        }
        SemanticDigest::from_bytes(w.finish())
    }
}

type Held = (SemanticDigest, SemanticDigest);

struct Working {
    numerical: BTreeSet<NumericalReason>,
    withdrawals: BTreeSet<Withdrawal>,
    holding: Option<Held>,
}

fn dependency_set(graph: &CompositionGraph, key: &str) -> BTreeSet<(String, DependencyKind)> {
    graph.dependencies(key).into_iter().map(|(up, kind)| (up.to_owned(), kind)).collect()
}

fn own_causes(
    previous: &CompositionGraph,
    current: &CompositionGraph,
    key: &str,
    effect: Option<&ObservedEffect>,
) -> Vec<NumericalCause> {
    let node = current.node(key).expect("key from the current graph");
    let mut causes = Vec::new();
    match previous.node(key) {
        None => causes.push(NumericalCause::Added),
        Some(before) => {
            let structure_changed = before.structure != node.structure;
            if structure_changed {
                causes.push(NumericalCause::StructureChanged);
            }
            if before.representation != node.representation {
                causes.push(NumericalCause::RepresentationChanged);
            }
            let coordinate_changed = before.coordinate != node.coordinate;
            if coordinate_changed {
                causes.push(NumericalCause::CoordinateChanged);
            }
            // A standing that moved without withdrawing is an own change: reusing the
            // stage artifact must not carry authority across it.
            if before.trust != node.trust
                && !matches!(node.trust, TrustStanding::Withdrawn | TrustStanding::Undeclared)
            {
                causes.push(NumericalCause::TrustChanged);
            }
            if matches!(before.support, SupportStanding::Unsupported { .. })
                && node.support == SupportStanding::Supported
            {
                causes.push(NumericalCause::SupportRestored);
            }
            if !coordinate_changed {
                if let (Some(was), Some(is)) =
                    (held_record(&before.calibration), held_record(&node.calibration))
                {
                    if was != is {
                        causes.push(NumericalCause::CalibrationChanged);
                    }
                }
            }
            if before.identity != node.identity && !structure_changed {
                // A node that carries a structural derivation and whose
                // structure is unchanged was refitted: the structure is
                // reusable, the old fit is not.
                causes.push(if node.structure.is_some() {
                    NumericalCause::StructureReusedFitNotRetained
                } else {
                    NumericalCause::IdentityChanged
                });
            } else if before.identity != node.identity {
                causes.push(NumericalCause::IdentityChanged);
            }
            if dependency_set(previous, key) != dependency_set(current, key) {
                // Dependencies on nodes that vanished are withdrawals, handled
                // separately; any other difference asks for recomputation.
                let kept: BTreeSet<_> = dependency_set(previous, key)
                    .into_iter()
                    .filter(|(up, _)| current.node(up).is_some())
                    .collect();
                if kept != dependency_set(current, key) {
                    causes.push(NumericalCause::DependenciesChanged);
                }
            }
        }
    }
    if matches!(effect, Some(ObservedEffect::Recomputed { .. })) {
        causes.push(NumericalCause::Observed);
    }
    causes
}

fn held_record(declaration: &CalibrationDeclaration) -> Option<Held> {
    match declaration {
        CalibrationDeclaration::Held { record, coordinate }
        | CalibrationDeclaration::Inherited { record, coordinate, .. } => {
            Some((*record, *coordinate))
        }
        CalibrationDeclaration::PointOnly | CalibrationDeclaration::Required => None,
    }
}

fn propagate_cause(kind: DependencyKind, cause: NumericalCause) -> NumericalCause {
    match (kind, cause) {
        (
            DependencyKind::StructuralDerivation,
            NumericalCause::IdentityChanged
            | NumericalCause::Observed
            | NumericalCause::Added
            | NumericalCause::CoordinateChanged
            | NumericalCause::DependenciesChanged
            | NumericalCause::RepresentationChanged
            | NumericalCause::TrustChanged
            | NumericalCause::CalibrationChanged
            | NumericalCause::SupportRestored,
        ) => NumericalCause::StructureReusedFitNotRetained,
        _ => cause,
    }
}

fn own_withdrawals(
    current: &CompositionGraph,
    key: &str,
    effect: Option<&ObservedEffect>,
) -> Vec<Withdrawal> {
    let node = current.node(key).expect("key from the current graph");
    let own = |kind, detail: String| Withdrawal { kind, origin: key.to_owned(), via: None, detail };
    let mut out = Vec::new();
    if let SupportStanding::Unsupported { coordinates } = &node.support {
        out.push(own(WithdrawalKind::Support, coordinates.join(",")));
    }
    if node.trust == TrustStanding::Withdrawn {
        out.push(own(WithdrawalKind::ProviderTrust, "provider_trust_withdrawn".to_owned()));
    }
    if node.trust == TrustStanding::Undeclared {
        out.push(own(WithdrawalKind::ProviderTrust, "provider_trust_undeclared".to_owned()));
    }
    match effect {
        Some(ObservedEffect::Withdrawn { cause }) => {
            out.push(own(WithdrawalKind::Retired, cause.clone()));
        }
        Some(ObservedEffect::Refused { detail }) => {
            out.push(own(WithdrawalKind::Unexecutable, detail.clone()));
        }
        _ => {}
    }
    out
}

fn evaluate_calibration(
    node: &super::graph::CompositionNode,
    blocked: bool,
    states: &BTreeMap<String, Working>,
) -> (CalibrationLineage, Option<Held>, Option<Withdrawal>) {
    let lost = |from: Option<String>, detail: &'static str| {
        (
            CalibrationLineage {
                state: CalibrationState::Lost,
                from,
                newly_required: true,
                detail: Some(detail),
            },
            None,
            Some(Withdrawal {
                kind: WithdrawalKind::Calibration,
                origin: node.key.clone(),
                via: None,
                detail: detail.to_owned(),
            }),
        )
    };
    match &node.calibration {
        CalibrationDeclaration::PointOnly => {
            (CalibrationLineage::plain(CalibrationState::NotApplicable), None, None)
        }
        CalibrationDeclaration::Required => (
            CalibrationLineage {
                state: CalibrationState::Required,
                from: None,
                newly_required: true,
                detail: Some("no_record"),
            },
            None,
            None,
        ),
        CalibrationDeclaration::Held { record, coordinate } => {
            if *coordinate != node.coordinate {
                return lost(None, "coordinate_mismatch");
            }
            let held = (!blocked).then_some((*record, *coordinate));
            (CalibrationLineage::plain(CalibrationState::Held), held, None)
        }
        CalibrationDeclaration::Inherited { from, record, coordinate } => {
            if *coordinate != node.coordinate {
                return lost(Some(from.clone()), "coordinate_mismatch");
            }
            let upstream = states.get(from).and_then(|s| s.holding);
            match upstream {
                None => lost(Some(from.clone()), "upstream_calibration_not_held"),
                Some(held) if held != (*record, *coordinate) => {
                    lost(Some(from.clone()), "record_differs")
                }
                Some(_) => (
                    CalibrationLineage {
                        state: CalibrationState::Inherited,
                        from: Some(from.clone()),
                        newly_required: false,
                        detail: None,
                    },
                    (!blocked).then_some((*record, *coordinate)),
                    None,
                ),
            }
        }
    }
}

/// Classify every claim in `current` against `previous`.
///
/// `effects` carries what the recalculation plan, the ledger or claim
/// freshness say about individual nodes. Keys not in `current` are ignored.
///
/// # Errors
/// A graph that does not validate (calibration inherited without a
/// calibration dependency, an unknown source node).
///
/// # Panics
/// Never for a graph that validates: every ordered key is a node of the graph.
pub fn revalidate(
    previous: &CompositionGraph,
    current: &CompositionGraph,
    effects: &BTreeMap<String, ObservedEffect>,
) -> Result<GraphRevalidation, GraphError> {
    current.validate()?;
    let mut states: BTreeMap<String, Working> = BTreeMap::new();
    let mut claims = BTreeMap::new();
    for key in current.topological_order() {
        let node = current.node(key).expect("ordered key is in the graph");
        let effect = effects.get(key);
        let mut working = Working {
            numerical: own_causes(previous, current, key, effect)
                .into_iter()
                .map(|cause| NumericalReason { origin: key.to_owned(), via: None, cause })
                .collect(),
            withdrawals: own_withdrawals(current, key, effect).into_iter().collect(),
            holding: None,
        };
        // Dependencies that vanished from the graph cannot be done without.
        for (up, kind) in dependency_set(previous, key) {
            if current.node(&up).is_none() {
                working.withdrawals.insert(Withdrawal {
                    kind: WithdrawalKind::Retired,
                    origin: up.clone(),
                    via: Some((up, kind)),
                    detail: "removed_from_graph".to_owned(),
                });
            }
        }
        for (up, kind) in current.dependencies(key) {
            let Some(upstream) = states.get(up) else { continue };
            for r in &upstream.numerical {
                // A replaced calibration record is a change only for claims that
                // claim to inherit calibration along a calibration dependency.
                if r.cause == NumericalCause::CalibrationChanged
                    && kind != DependencyKind::Calibration
                {
                    continue;
                }
                working.numerical.insert(NumericalReason {
                    origin: r.origin.clone(),
                    via: Some((up.to_owned(), kind)),
                    cause: propagate_cause(kind, r.cause),
                });
            }
            for w in &upstream.withdrawals {
                if w.kind == WithdrawalKind::Calibration && kind != DependencyKind::Calibration {
                    continue;
                }
                working.withdrawals.insert(Withdrawal {
                    kind: w.kind,
                    origin: w.origin.clone(),
                    via: Some((up.to_owned(), kind)),
                    detail: w.detail.clone(),
                });
            }
        }
        let blocked = working.withdrawals.iter().any(|w| w.kind != WithdrawalKind::Calibration);
        let (calibration, holding, withdrawal) = evaluate_calibration(node, blocked, &states);
        working.holding = holding;
        working.withdrawals.extend(withdrawal);
        let class = if !working.withdrawals.is_empty() {
            Revalidation::Withdrawn
        } else if !working.numerical.is_empty() {
            Revalidation::Recompute
        } else {
            Revalidation::StillValid
        };
        claims.insert(
            key.to_owned(),
            ClaimRevalidation {
                key: key.to_owned(),
                kind: node.kind,
                class,
                numerical: working.numerical.iter().cloned().collect(),
                withdrawals: working.withdrawals.iter().cloned().collect(),
                authority: if class == Revalidation::StillValid {
                    PriorAuthority::Retained
                } else {
                    PriorAuthority::NotRetained
                },
                calibration,
            },
        );
        states.insert(key.to_owned(), working);
    }
    let removed = previous
        .nodes()
        .filter(|n| current.node(&n.key).is_none())
        .map(|n| n.key.clone())
        .collect();
    Ok(GraphRevalidation {
        claims,
        removed,
        previous: previous.identity(),
        current: current.identity(),
    })
}
