//! Portable claim envelope over a compiled contract and one result.
//!
//! A program defines an analysis. A claim records what a particular
//! execution or licensed derivation concludes. Receiving bytes,
//! understanding semantics, verifying dependencies, and being able to
//! execute remain separate states.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::identity::{ContractIdentities, SemanticDigest};
use crate::provenance::{ProvenanceGraph, ProvenanceNode};
use crate::reasoning::ReasoningView;

/// Kind of value a claim carries. Refusals and bounds are first-class.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum ClaimKind {
    /// Scalar point estimate.
    Point,
    /// Identified-set bounds.
    Bounds,
    /// Probability-weighted mixture.
    Mixture,
    /// Function-valued response.
    Response,
    /// Structured refusal.
    Refusal,
    /// Incomplete contract with explicit obligations.
    Incomplete,
}

impl ClaimKind {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Point => "point",
            Self::Bounds => "bounds",
            Self::Mixture => "mixture",
            Self::Response => "response",
            Self::Refusal => "refusal",
            Self::Incomplete => "incomplete",
        }
    }
}

/// Status of a requested coordinate relative to a claim.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum DomainStatus {
    /// Identified at this coordinate.
    Identified,
    /// Empirically supported at this coordinate.
    Supported,
    /// Actually evaluated at this coordinate.
    Evaluated,
    /// Outside the declared domain.
    OutsideScope,
    /// Unknown; not a failed support and not an executable guarantee.
    Unknown,
}

impl DomainStatus {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Identified => "identified",
            Self::Supported => "supported",
            Self::Evaluated => "evaluated",
            Self::OutsideScope => "outside_scope",
            Self::Unknown => "unknown",
        }
    }
}

/// Separate identification / support / evaluation domains.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct ClaimDomains {
    /// Identification domain status.
    pub identification: DomainStatus,
    /// Empirical support domain status.
    pub support: DomainStatus,
    /// Actually evaluated domain status.
    pub evaluated: DomainStatus,
}

impl ClaimDomains {
    /// Construct the three domain statuses.
    #[must_use]
    pub const fn new(
        identification: DomainStatus,
        support: DomainStatus,
        evaluated: DomainStatus,
    ) -> Self {
        Self { identification, support, evaluated }
    }
}

/// Versioned claim envelope. Scientific payload encodings stay in `antecedent-io`.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct ClaimEnvelope {
    /// Claim id (content-addressed from the envelope fields).
    pub claim_id: SemanticDigest,
    /// Program / target / inference identities this claim references.
    pub identities: ContractIdentities,
    /// Result kind.
    pub kind: ClaimKind,
    /// Scalar value when [`ClaimKind::Point`]; `None` is explicit absence.
    pub value: Option<f64>,
    /// Outcome scale / units, when known.
    pub outcome_units: Option<Arc<str>>,
    /// Four reasoning slots.
    pub reasoning: ReasoningView,
    /// Domain statuses.
    pub domains: ClaimDomains,
    /// Producer / execution lineage digest, when an execution exists.
    pub execution: Option<SemanticDigest>,
    /// Evidence / dependency references (artifact ids, fixture names).
    pub evidence: Arc<[Arc<str>]>,
}

impl ClaimEnvelope {
    /// Construct a claim envelope.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        claim_id: SemanticDigest,
        identities: ContractIdentities,
        kind: ClaimKind,
        value: Option<f64>,
        outcome_units: Option<Arc<str>>,
        reasoning: ReasoningView,
        domains: ClaimDomains,
        execution: Option<SemanticDigest>,
        evidence: impl Into<Arc<[Arc<str>]>>,
    ) -> Self {
        Self {
            claim_id,
            identities,
            kind,
            value,
            outcome_units,
            reasoning,
            domains,
            execution,
            evidence: evidence.into(),
        }
    }
}

/// Receiving-system acceptance report. Distinct from byte storage.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct AcceptanceReport {
    /// Whether required semantic features were recognized.
    pub recognized: bool,
    /// Whether referenced identities / sections verified.
    pub verified_references: bool,
    /// Unresolved dependencies.
    pub unresolved: Arc<[Arc<str>]>,
    /// Operations the receiver can perform.
    pub supported_operations: Arc<[Arc<str>]>,
    /// Why acceptance is restricted or refused.
    pub restriction: Option<Arc<str>>,
}

impl AcceptanceReport {
    /// Construct a receiving-system report.
    #[must_use]
    pub fn new(
        recognized: bool,
        verified_references: bool,
        unresolved: impl Into<Arc<[Arc<str>]>>,
        supported_operations: impl Into<Arc<[Arc<str>]>>,
        restriction: Option<Arc<str>>,
    ) -> Self {
        Self {
            recognized,
            verified_references,
            unresolved: unresolved.into(),
            supported_operations: supported_operations.into(),
            restriction,
        }
    }

    /// Opaque storage/forwarding without interpretation.
    #[must_use]
    pub fn opaque_storage() -> Self {
        Self {
            recognized: false,
            verified_references: false,
            unresolved: Arc::from([Arc::from("required_semantics")]),
            supported_operations: Arc::from([Arc::from("store"), Arc::from("forward")]),
            restriction: Some(Arc::from("opaque_envelope")),
        }
    }

    /// Whether this report accepts the claim as a usable causal claim.
    #[must_use]
    pub fn accepts_as_claim(&self) -> bool {
        self.recognized && self.verified_references && self.unresolved.is_empty()
    }

    /// Whether the artifact is a fully verified program or claim.
    ///
    /// Distinct from storage/forwarding. Old artifacts without a contract
    /// section remain readable but do not pass this check.
    #[must_use]
    pub fn accepts_as_verified_program(&self) -> bool {
        self.accepts_as_claim()
    }
}

/// Handoff / loss receipt. Chained receipts accumulate unresolved losses.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct HandoffReceipt {
    /// Input claim id.
    pub input: SemanticDigest,
    /// Output claim id, when a claim was produced.
    pub output: Option<SemanticDigest>,
    /// Consumer capability profile id.
    pub consumer: Arc<str>,
    /// Transformation rule applied.
    pub rule: Arc<str>,
    /// Fields / sections retained.
    pub retained: Arc<[Arc<str>]>,
    /// Fields / sections omitted.
    pub omitted: Arc<[Arc<str>]>,
    /// Unresolved references after the handoff.
    pub unresolved: Arc<[Arc<str>]>,
    /// Operations no longer available.
    pub unavailable_operations: Arc<[Arc<str>]>,
}

impl HandoffReceipt {
    /// Construct a handoff / loss receipt.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: SemanticDigest,
        output: Option<SemanticDigest>,
        consumer: impl Into<Arc<str>>,
        rule: impl Into<Arc<str>>,
        retained: impl Into<Arc<[Arc<str>]>>,
        omitted: impl Into<Arc<[Arc<str>]>>,
        unresolved: impl Into<Arc<[Arc<str>]>>,
        unavailable_operations: impl Into<Arc<[Arc<str>]>>,
    ) -> Self {
        Self {
            input,
            output,
            consumer: consumer.into(),
            rule: rule.into(),
            retained: retained.into(),
            omitted: omitted.into(),
            unresolved: unresolved.into(),
            unavailable_operations: unavailable_operations.into(),
        }
    }

    /// Lossless exchange: every required field retained, output is the same claim.
    #[must_use]
    pub fn lossless(input: SemanticDigest, consumer: impl Into<Arc<str>>) -> Self {
        Self::new(
            input,
            Some(input),
            consumer,
            "lossless_exchange",
            [
                Arc::from("claim.envelope"),
                Arc::from("identities"),
                Arc::from("reasoning"),
                Arc::from("domains"),
            ],
            [],
            [],
            [],
        )
    }

    /// Lossy scalar view: may reference a complete claim, cannot impersonate it.
    #[must_use]
    pub fn lossy_scalar(input: SemanticDigest, consumer: impl Into<Arc<str>>) -> Self {
        Self::new(
            input,
            Some(input),
            consumer,
            "scalar_view",
            [Arc::from("value"), Arc::from("claim.id")],
            [
                Arc::from("unidentified_mass"),
                Arc::from("reasoning"),
                Arc::from("domains"),
                Arc::from("identities"),
            ],
            [],
            [Arc::from("accept_as_claim"), Arc::from("re_estimate")],
        )
    }

    /// Opaque storage / forward without interpreting required semantics.
    #[must_use]
    pub fn opaque_forward(input: SemanticDigest, consumer: impl Into<Arc<str>>) -> Self {
        Self::new(
            input,
            Some(input),
            consumer,
            "opaque_forward",
            [Arc::from("bytes")],
            [Arc::from("required_semantics")],
            [Arc::from("required_semantics")],
            [
                Arc::from("accept_as_claim"),
                Arc::from("inspect_claim"),
                Arc::from("verify_contract"),
            ],
        )
    }

    /// Whether omitted required meaning prevents equivalent-claim acceptance.
    #[must_use]
    pub fn equivalent_claim(&self) -> bool {
        self.omitted.is_empty() && self.unresolved.is_empty() && self.output.is_some()
    }

    /// Accumulate unresolved losses from `self` then `next`.
    #[must_use]
    pub fn chain(&self, next: &Self) -> Self {
        Self {
            input: self.input,
            output: next.output,
            consumer: next.consumer.clone(),
            rule: next.rule.clone(),
            retained: next.retained.clone(),
            omitted: merge_unique(&self.omitted, &next.omitted),
            unresolved: merge_unique(&self.unresolved, &next.unresolved),
            unavailable_operations: next.unavailable_operations.clone(),
        }
    }
}

fn merge_unique(left: &[Arc<str>], right: &[Arc<str>]) -> Arc<[Arc<str>]> {
    let mut out: Vec<Arc<str>> = left.to_vec();
    for item in right {
        if !out.iter().any(|existing| existing == item) {
            out.push(item.clone());
        }
    }
    Arc::from(out)
}

/// Which of the three claim domains is being asked about.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum ClaimDomainAxis {
    /// Identification domain.
    Identification,
    /// Empirical support domain.
    Support,
    /// Actually evaluated domain.
    Evaluated,
}

impl ClaimDomainAxis {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Identification => "identification",
            Self::Support => "support",
            Self::Evaluated => "evaluated",
        }
    }
}

/// Licensed composition that may produce a derived claim. Comparison is not synthesis.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum ClaimOperation {
    /// Expose differences without pooling.
    Compare,
    /// Contrast requiring a licensed joint uncertainty contract.
    Contrast,
    /// Aggregate requiring a licensed joint inference contract.
    Aggregate,
    /// Retarget within an existing certified derivation.
    Retarget,
    /// Pool / synthesize. Unlicensed in 1.10; not transport.
    Pool,
}

impl ClaimOperation {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Compare => "compare",
            Self::Contrast => "contrast",
            Self::Aggregate => "aggregate",
            Self::Retarget => "retarget",
            Self::Pool => "pool",
        }
    }

    /// Whether this operation synthesizes a new licensed claim.
    #[must_use]
    pub const fn synthesizes(self) -> bool {
        matches!(self, Self::Contrast | Self::Aggregate | Self::Pool)
    }
}

/// Compatibility report for one claim operation. Matching names are insufficient.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct ClaimCompatibility {
    /// Requested operation.
    pub operation: ClaimOperation,
    /// Whether the operation is licensed for these parents.
    pub compatible: bool,
    /// Unresolved obligations (alignment, mapping, identity disagreement).
    pub unresolved: Arc<[Arc<str>]>,
    /// Why the operation is restricted or refused.
    pub restriction: Option<Arc<str>>,
}

impl ClaimCompatibility {
    /// Construct a compatibility report.
    #[must_use]
    pub fn new(
        operation: ClaimOperation,
        compatible: bool,
        unresolved: impl Into<Arc<[Arc<str>]>>,
        restriction: Option<Arc<str>>,
    ) -> Self {
        Self { operation, compatible, unresolved: unresolved.into(), restriction }
    }
}

/// Declared dependence between evidence sources. Unknown is explicit.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum EvidenceDependence {
    /// Distinct snapshots with a declared independence.
    Independent,
    /// Shared snapshot, units, draws, or forwarded duplicate.
    Shared,
    /// Dependence not declared. Not an independence default.
    Unknown,
}

impl EvidenceDependence {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Independent => "independent",
            Self::Shared => "shared",
            Self::Unknown => "unknown",
        }
    }
}

/// Shared-evidence identity without raw-data disclosure.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct SharedEvidenceRef {
    /// Data-snapshot identity.
    pub snapshot: SemanticDigest,
    /// Identification / graph-origin identity.
    pub graph_origin: SemanticDigest,
    /// Prior-origin label, when a mapping exists.
    pub prior_origin: Option<Arc<str>>,
    /// Joint-draw / covariance / replicate alignment, when declared.
    pub alignment: Option<Arc<str>>,
    /// Dependence declaration. Forwarding one claim twice is [`EvidenceDependence::Shared`].
    pub dependence: EvidenceDependence,
}

impl SharedEvidenceRef {
    /// Project evidence identity from a claim's existing identity layers.
    #[must_use]
    pub fn from_claim(claim: &ClaimEnvelope) -> Self {
        Self {
            snapshot: claim.identities.data_snapshot,
            graph_origin: claim.identities.identification,
            prior_origin: claim.evidence.iter().find(|item| item.starts_with("prior:")).cloned(),
            alignment: claim
                .evidence
                .iter()
                .find(|item| {
                    item.starts_with("alignment:")
                        || matches!(&***item, "joint_draws" | "covariance" | "replicate_ids")
                })
                .cloned(),
            dependence: EvidenceDependence::Unknown,
        }
    }

    /// Same snapshot or same claim id is one source, not two.
    #[must_use]
    pub fn same_source(&self, other: &Self) -> bool {
        self.snapshot == other.snapshot || self.graph_origin == other.graph_origin
    }

    /// Forwarding the same claim id is a shared source, never independence.
    #[must_use]
    pub fn forwarded_duplicate(left: SemanticDigest, right: SemanticDigest) -> bool {
        left == right
    }
}

/// Freshness as of a receiver's last verified snapshot. Timestamps do not license.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct ClaimFreshness {
    /// The claim was valid for the snapshot that produced it.
    pub historically_valid: bool,
    /// `Some` when the receiver has a current snapshot; `None` when offline.
    pub currently_applicable: Option<bool>,
    /// Local runtime readiness. Distinct from scientific applicability.
    pub locally_ready: bool,
    /// Replacement claim, when recorded.
    pub superseded_by: Option<SemanticDigest>,
    /// Explicit withdrawal. Original claim is preserved.
    pub withdrawn: bool,
    /// Snapshot the receiver last verified. Offline reports this, not "current".
    pub as_of_snapshot: Option<SemanticDigest>,
}

impl ClaimFreshness {
    /// Bind freshness to versioned dependencies. No timestamp argument.
    #[must_use]
    pub fn of(
        claim: &ClaimEnvelope,
        last_verified_snapshot: Option<SemanticDigest>,
        current_snapshot: Option<SemanticDigest>,
        superseded_by: Option<SemanticDigest>,
        withdrawn: bool,
        locally_ready: bool,
    ) -> Self {
        let currently_applicable = current_snapshot.map(|snapshot| {
            snapshot == claim.identities.data_snapshot && superseded_by.is_none() && !withdrawn
        });
        Self {
            historically_valid: true,
            currently_applicable,
            locally_ready,
            superseded_by,
            withdrawn,
            as_of_snapshot: last_verified_snapshot,
        }
    }
}

/// Consumer capability profile. Storage/forwarding is not claim acceptance.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct ConsumerProfile {
    /// Stable profile id (`full`, `restricted`, `forwarding`).
    pub id: Arc<str>,
    /// Semantic features this consumer can interpret.
    pub features: Arc<[Arc<str>]>,
}

impl ConsumerProfile {
    /// Sender / full receiver: current contract + every claim kind.
    #[must_use]
    pub fn full() -> Self {
        Self::with_features(
            "full",
            &[
                "store",
                "forward",
                "read_body",
                "verify_contract",
                "inspect_claim",
                "contract.v1",
                "claim.point",
                "claim.bounds",
                "claim.mixture",
                "claim.response",
                "claim.refusal",
                "claim.incomplete",
                "reasoning.four_slots",
            ],
        )
    }

    /// Restricted consumer: point claims only. Unknown required features refuse acceptance.
    #[must_use]
    pub fn restricted() -> Self {
        Self::with_features(
            "restricted",
            &[
                "store",
                "forward",
                "read_body",
                "verify_contract",
                "inspect_claim",
                "contract.v1",
                "claim.point",
                "reasoning.four_slots",
            ],
        )
    }

    /// Forwarding consumer: byte storage only. Cannot accept a usable claim.
    #[must_use]
    pub fn forwarding() -> Self {
        Self::with_features("forwarding", &["store", "forward"])
    }

    fn with_features(id: &'static str, features: &[&'static str]) -> Self {
        Self { id: Arc::from(id), features: features.iter().copied().map(Arc::from).collect() }
    }

    /// Whether this profile interprets `feature`.
    #[must_use]
    pub fn understands(&self, feature: &str) -> bool {
        self.features.iter().any(|known| &**known == feature)
    }
}

/// Host operation names. Adapters call existing facade / consume seams.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum HostOperation {
    /// Cheap inspect. Does not identify.
    Inspect,
    /// Accept / verify a claim from bytes.
    AcceptClaim,
    /// Ask whether a coordinate is identified / supported / evaluated.
    CheckDomain,
    /// Compare / contrast / aggregate / retarget compatibility.
    CheckCompatibility,
    /// Transformation preview.
    PreviewTransform,
    /// Checked transformation apply.
    ApplyTransform,
    /// Capability / blocker report.
    ExplainBlockers,
    /// Execute against a bound snapshot.
    Execute,
    /// Export a contracted artifact.
    Export,
}

impl HostOperation {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Inspect => "inspect",
            Self::AcceptClaim => "accept_claim",
            Self::CheckDomain => "check_domain",
            Self::CheckCompatibility => "check_compatibility",
            Self::PreviewTransform => "preview_transform",
            Self::ApplyTransform => "apply_transform",
            Self::ExplainBlockers => "explain_blockers",
            Self::Execute => "execute",
            Self::Export => "export",
        }
    }
}

/// Derived claim recorded on the existing [`ProvenanceGraph`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct DerivedClaim {
    /// Derived envelope. Parents are not retroactively strengthened.
    pub claim: ClaimEnvelope,
    /// Parent claim ids.
    pub parents: Arc<[SemanticDigest]>,
    /// Transformation / operation identity.
    pub operation: ClaimOperation,
    /// Ancestry. Unique ids, no cycles; unresolved external parents are recorded.
    pub provenance: ProvenanceGraph,
    /// Derivation receipt.
    pub receipt: HandoffReceipt,
}

/// Outcome of a claimed composition. Unlicensed synthesis keeps parents intact.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum DerivedClaimOutcome {
    /// Licensed derivation with parent receipts.
    Derived(Box<DerivedClaim>),
    /// Comparison-only: differences without a pooled claim.
    Comparison(ClaimCompatibility),
    /// Typed refusal. Parent claim ids are unchanged.
    Refused {
        /// Why the composition is refused.
        restriction: Arc<str>,
        /// Parent claim ids, intact.
        parents: Arc<[SemanticDigest]>,
        /// Compatibility report when one was computed.
        compatibility: Option<ClaimCompatibility>,
    },
}

impl ClaimEnvelope {
    /// Status of one domain axis.
    #[must_use]
    pub const fn domain(&self, axis: ClaimDomainAxis) -> DomainStatus {
        match axis {
            ClaimDomainAxis::Identification => self.domains.identification,
            ClaimDomainAxis::Support => self.domains.support,
            ClaimDomainAxis::Evaluated => self.domains.evaluated,
        }
    }

    /// Shared-evidence identity projected from existing layers.
    #[must_use]
    pub fn evidence_ref(&self) -> SharedEvidenceRef {
        SharedEvidenceRef::from_claim(self)
    }

    /// Freshness as of the receiver's last verified snapshot.
    #[must_use]
    pub fn freshness(
        &self,
        last_verified_snapshot: Option<SemanticDigest>,
        current_snapshot: Option<SemanticDigest>,
        superseded_by: Option<SemanticDigest>,
        withdrawn: bool,
        locally_ready: bool,
    ) -> ClaimFreshness {
        ClaimFreshness::of(
            self,
            last_verified_snapshot,
            current_snapshot,
            superseded_by,
            withdrawn,
            locally_ready,
        )
    }

    /// Compatibility of `self` and `other` for `operation`.
    #[must_use]
    pub fn compatibility(&self, other: &Self, operation: ClaimOperation) -> ClaimCompatibility {
        claim_compatibility(&[self, other], operation, None)
    }
}

/// Compatibility for a parent set. Identity disagreement and missing alignment refuse.
#[must_use]
pub fn claim_compatibility(
    parents: &[&ClaimEnvelope],
    operation: ClaimOperation,
    alignment: Option<&str>,
) -> ClaimCompatibility {
    if parents.is_empty() {
        return ClaimCompatibility::new(
            operation,
            false,
            [Arc::from("parents")],
            Some(Arc::from("need_parent_claims")),
        );
    }
    if parents.len() < 2 && operation != ClaimOperation::Retarget {
        return ClaimCompatibility::new(
            operation,
            false,
            [Arc::from("parents")],
            Some(Arc::from("need_two_claims")),
        );
    }
    let mut unresolved = Vec::new();
    let first = parents[0];
    for parent in parents.iter().skip(1) {
        if parent.identities.observation != first.identities.observation {
            unresolved.push(Arc::from("observation"));
        }
        if parent.outcome_units != first.outcome_units {
            unresolved.push(Arc::from("outcome_units"));
        }
    }
    match operation {
        ClaimOperation::Compare => {
            ClaimCompatibility::new(operation, unresolved.is_empty(), unresolved, None)
        }
        ClaimOperation::Retarget => {
            let product = first.identities.identification_product;
            let same_product = product.is_some()
                && parents.iter().all(|parent| parent.identities.identification_product == product);
            if !same_product {
                unresolved.push(Arc::from("identification_product"));
            }
            let restriction =
                (!same_product).then(|| Arc::from("retarget_requires_shared_product"));
            ClaimCompatibility::new(operation, unresolved.is_empty(), unresolved, restriction)
        }
        ClaimOperation::Contrast | ClaimOperation::Aggregate => {
            if !alignment_is_licensed(parents, alignment) {
                unresolved.push(Arc::from("alignment"));
            }
            let restriction = unresolved
                .iter()
                .any(|item| &**item == "alignment")
                .then(|| Arc::from("marginal_intervals_do_not_determine_contrast"));
            ClaimCompatibility::new(operation, unresolved.is_empty(), unresolved, restriction)
        }
        ClaimOperation::Pool => ClaimCompatibility::new(
            operation,
            false,
            [Arc::from("transport")],
            Some(Arc::from("unlicensed_synthesis")),
        ),
    }
}

fn alignment_is_licensed(parents: &[&ClaimEnvelope], alignment: Option<&str>) -> bool {
    let declared = alignment.map(str::to_owned).or_else(|| {
        parents.iter().find_map(|parent| parent.evidence_ref().alignment.map(|a| a.to_string()))
    });
    matches!(
        declared.as_deref(),
        Some(
            "joint_draws"
                | "covariance"
                | "replicate_ids"
                | "alignment:joint_draws"
                | "alignment:covariance"
                | "alignment:replicate_ids"
        )
    )
}

/// Compose parent claims. Reuses [`ProvenanceGraph`]; does not invent a verifier.
#[must_use]
pub fn compose_claims(
    parents: &[&ClaimEnvelope],
    operation: ClaimOperation,
    derived: Option<ClaimEnvelope>,
    alignment: Option<&str>,
) -> DerivedClaimOutcome {
    let parent_ids: Arc<[SemanticDigest]> = parents.iter().map(|parent| parent.claim_id).collect();
    if operation == ClaimOperation::Compare {
        return DerivedClaimOutcome::Comparison(claim_compatibility(parents, operation, alignment));
    }
    let compatibility = claim_compatibility(parents, operation, alignment);
    if !compatibility.compatible {
        return refuse_derived(
            compatibility.restriction.clone().unwrap_or_else(|| Arc::from("incompatible_claims")),
            parent_ids,
            Some(compatibility),
        );
    }
    let Some(derived) = derived else {
        return refuse_derived("missing_derived_envelope", parent_ids, Some(compatibility));
    };
    if parents_strengthened(parents, &derived) {
        return refuse_derived(
            "parents_not_retroactively_strengthened",
            parent_ids,
            Some(compatibility),
        );
    }
    let mut provenance = ProvenanceGraph::new();
    for parent in parents {
        let node = provenance_node(parent.claim_id, operation.as_str(), &[]);
        if provenance.try_push(node).is_err() {
            return refuse_derived("provenance_parent", parent_ids, Some(compatibility));
        }
    }
    let parent_hex: Vec<Arc<str>> =
        parents.iter().map(|parent| Arc::from(parent.claim_id.to_hex())).collect();
    if provenance
        .try_push(provenance_node(derived.claim_id, operation.as_str(), &parent_hex))
        .is_err()
    {
        return refuse_derived("provenance_cycle_or_duplicate", parent_ids, Some(compatibility));
    }
    DerivedClaimOutcome::Derived(Box::new(DerivedClaim {
        receipt: HandoffReceipt::new(
            parents[0].claim_id,
            Some(derived.claim_id),
            "derive",
            operation.as_str(),
            [Arc::from("claim.envelope"), Arc::from("parents"), Arc::from("provenance")],
            [],
            provenance
                .validate()
                .map_or_else(|_| vec![Arc::from("provenance")], |missing| missing.to_vec()),
            [],
        ),
        claim: derived,
        parents: parent_ids,
        operation,
        provenance,
    }))
}

fn refuse_derived(
    restriction: impl Into<Arc<str>>,
    parents: Arc<[SemanticDigest]>,
    compatibility: Option<ClaimCompatibility>,
) -> DerivedClaimOutcome {
    DerivedClaimOutcome::Refused { restriction: restriction.into(), parents, compatibility }
}

fn parents_strengthened(parents: &[&ClaimEnvelope], derived: &ClaimEnvelope) -> bool {
    let Some(derived_mass) = identified_mass(derived) else {
        return false;
    };
    parents
        .iter()
        .any(|parent| identified_mass(parent).is_some_and(|mass| derived_mass > mass + 1e-12))
}

fn identified_mass(claim: &ClaimEnvelope) -> Option<f64> {
    claim.reasoning.identification.as_ref().map(|slot| slot.identified_mass)
}

fn provenance_node(
    claim_id: SemanticDigest,
    operation: &str,
    parents: &[Arc<str>],
) -> ProvenanceNode {
    ProvenanceNode {
        artifact_id: Arc::from(claim_id.to_hex()),
        operation: Arc::from(operation),
        parents: parents.iter().cloned().collect(),
        assumptions: crate::assumption::AssumptionSet::new(),
        library_version: Arc::from(crate::VERSION),
        config_digest: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> SemanticDigest {
        let mut bytes = [0u8; 32];
        bytes[0] = byte;
        SemanticDigest::from_bytes(bytes)
    }

    #[test]
    fn opaque_storage_is_not_claim_acceptance() {
        let report = AcceptanceReport::opaque_storage();
        assert!(!report.accepts_as_claim());
        assert!(report.supported_operations.iter().any(|op| &**op == "forward"));
    }

    #[test]
    fn chained_receipts_accumulate_losses() {
        let first = HandoffReceipt {
            input: digest(1),
            output: Some(digest(2)),
            consumer: Arc::from("restricted"),
            rule: Arc::from("scalar_view"),
            retained: Arc::from([Arc::from("value")]),
            omitted: Arc::from([Arc::from("unidentified_mass")]),
            unresolved: Arc::from([]),
            unavailable_operations: Arc::from([Arc::from("re_estimate")]),
        };
        let second = HandoffReceipt {
            input: digest(2),
            output: Some(digest(3)),
            consumer: Arc::from("forward"),
            rule: Arc::from("reserialize"),
            retained: Arc::from([Arc::from("value")]),
            omitted: Arc::from([]),
            unresolved: Arc::from([Arc::from("covariance")]),
            unavailable_operations: Arc::from([]),
        };
        let chained = first.chain(&second);
        assert!(!chained.equivalent_claim());
        assert!(chained.omitted.iter().any(|f| &**f == "unidentified_mass"));
        assert!(chained.unresolved.iter().any(|f| &**f == "covariance"));
        let recovered = chained.chain(&HandoffReceipt::lossless(digest(3), "reserialize"));
        assert!(!recovered.equivalent_claim());
        assert!(recovered.omitted.iter().any(|f| &**f == "unidentified_mass"));
    }

    #[test]
    fn lossy_scalar_cannot_impersonate_a_complete_claim() {
        let view = HandoffReceipt::lossy_scalar(digest(1), "restricted");
        assert!(!view.equivalent_claim());
        assert!(view.omitted.iter().any(|field| &**field == "unidentified_mass"));
    }

    #[test]
    fn forwarded_duplicate_is_not_two_sources() {
        let id = digest(9);
        assert!(SharedEvidenceRef::forwarded_duplicate(id, id));
        assert!(!SharedEvidenceRef::forwarded_duplicate(id, digest(8)));
    }

    #[test]
    fn unlicensed_pool_refuses_with_parents_intact() {
        let left = sample_claim(1, 0.4);
        let right = sample_claim(2, 0.4);
        match compose_claims(&[&left, &right], ClaimOperation::Pool, None, None) {
            DerivedClaimOutcome::Refused { restriction, parents, .. } => {
                assert_eq!(&*restriction, "unlicensed_synthesis");
                assert_eq!(parents.as_ref(), [left.claim_id, right.claim_id]);
            }
            other => panic!("expected refusal, got {other:?}"),
        }
        match compose_claims(&[&left, &right], ClaimOperation::Contrast, None, None) {
            DerivedClaimOutcome::Refused { restriction, .. } => {
                assert_eq!(&*restriction, "marginal_intervals_do_not_determine_contrast");
            }
            other => panic!("expected contrast refusal, got {other:?}"),
        }
        match compose_claims(&[&left, &right], ClaimOperation::Compare, None, None) {
            DerivedClaimOutcome::Comparison(report) => {
                assert!(report.compatible);
                assert!(!report.operation.synthesizes());
            }
            other => panic!("expected comparison, got {other:?}"),
        }
    }

    #[test]
    fn retarget_derived_claim_reuses_provenance_and_refuses_stronger_child() {
        let parent = sample_claim(1, 0.4);
        match compose_claims(&[&parent], ClaimOperation::Retarget, Some(sample_claim(3, 0.4)), None)
        {
            DerivedClaimOutcome::Derived(record) => {
                assert_eq!(record.parents.as_ref(), [parent.claim_id]);
                assert!(record.receipt.equivalent_claim());
                assert!(record.provenance.validate().unwrap().is_empty());
            }
            other => panic!("expected derived retarget, got {other:?}"),
        }
        match compose_claims(&[&parent], ClaimOperation::Retarget, Some(sample_claim(4, 1.0)), None)
        {
            DerivedClaimOutcome::Refused { restriction, parents, .. } => {
                assert_eq!(&*restriction, "parents_not_retroactively_strengthened");
                assert_eq!(parents.as_ref(), [parent.claim_id]);
            }
            other => panic!("expected stronger-child refusal, got {other:?}"),
        }
    }

    fn sample_claim(byte: u8, identified_mass: f64) -> ClaimEnvelope {
        use crate::identification::IdentificationStatus;
        use crate::reasoning::{
            AssumptionSlot, IdentificationSlot, ReasoningView, SlotAvailability, SupportSlot,
            UncertaintySlot,
        };
        ClaimEnvelope::new(
            digest(byte),
            ContractIdentities::new(
                digest(10),
                digest(11),
                Some(digest(12)),
                digest(13),
                digest(14),
                digest(15),
                digest(16),
            ),
            ClaimKind::Point,
            Some(0.0),
            None,
            ReasoningView::new(
                SlotAvailability::Available(IdentificationSlot::new(
                    IdentificationStatus::NonparametricallyIdentified,
                    identified_mass,
                    1.0 - identified_mass,
                    0.0,
                    0.0,
                    true,
                    None,
                    false,
                )),
                SlotAvailability::Available(SupportSlot::new(
                    "licensed",
                    None,
                    SlotAvailability::unavailable("not_evaluated"),
                )),
                SlotAvailability::Available(UncertaintySlot::new([])),
                SlotAvailability::Available(AssumptionSlot::new([])),
            ),
            ClaimDomains::new(
                DomainStatus::Identified,
                DomainStatus::Unknown,
                DomainStatus::Evaluated,
            ),
            Some(digest(20)),
            [Arc::from("snapshot:test")],
        )
    }
}
