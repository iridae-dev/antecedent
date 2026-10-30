//! Portable claim envelope over a compiled contract and one result.
//!
//! A program defines an analysis. A claim records what a particular
//! execution or licensed derivation concludes. Receiving bytes,
//! understanding semantics, verifying dependencies, and being able to
//! execute remain separate states.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::provenance::{ProvenanceGraph, ProvenanceNode};
use crate::reasoning::ReasoningView;
use antecedent_core::identity::CanonicalWriter;
use antecedent_core::named_variants;
use antecedent_core::scientific_target::{ScientificTarget, TargetForm};
use antecedent_core::{CalibrationBasis, ContractIdentities, SemanticDigest};

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
    /// Structured held-out policy-value comparison (not a scalar treatment effect).
    PolicyValue,
    /// Structured refusal.
    Refusal,
    /// Incomplete contract with explicit obligations.
    Incomplete,
    /// Noncausal association, regression or prediction. Identification is not
    /// claimed, so the claim never reads as an intervention effect.
    Associational,
}

named_variants! {
    ClaimKind {
        Point => "point",
        Bounds => "bounds",
        Mixture => "mixture",
        Response => "response",
        PolicyValue => "policy_value",
        Refusal => "refusal",
        Incomplete => "incomplete",
        Associational => "associational",
    }
}

impl ClaimKind {
    /// Whether the kind states something about an intervention.
    ///
    /// An associational claim describes the observed joint law and nothing
    /// more; it ranks no action and is never counted as an effect.
    #[must_use]
    pub const fn is_causal(self) -> bool {
        !matches!(self, Self::Associational)
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
    /// An empirical check at this coordinate ran and failed.
    Contradicted,
    /// Unknown; not a failed support and not an executable guarantee.
    Unknown,
    /// Supported only under a declared adjustment that changed what the answer covers or how
    /// it weights the data: an overlap clip that binds, or a trim that excluded rows. The
    /// claim holds for that adjusted target; a host relays the limit with the answer.
    Limited,
    /// Identification is not claimed: the claim is noncausal by construction,
    /// so there is nothing to identify. Not a failed or missing identification.
    NotClaimed,
}

named_variants! {
    DomainStatus {
        Identified => "identified",
        Supported => "supported",
        Evaluated => "evaluated",
        OutsideScope => "outside_scope",
        Contradicted => "contradicted",
        Unknown => "unknown",
        Limited => "limited",
        NotClaimed => "not_claimed",
    }
}

impl DomainStatus {
    /// Whether the status asserts something (identified, supported, evaluated).
    #[must_use]
    const fn is_positive(self) -> bool {
        matches!(self, Self::Identified | Self::Supported | Self::Evaluated)
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
    /// Calibration slot recorded at claim time.
    pub calibration: CalibrationView,
    /// Attested custom-validator evidence.
    pub attested: Arc<[AttestedEvidence]>,
}

/// Calibration slot projected onto a claim.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationView {
    /// `calibrated` | `scope_not_assessed` | `unavailable`.
    pub status: Arc<str>,
    /// Governing coverage record id.
    pub record_id: Option<Arc<str>>,
    /// Reason code when not calibrated.
    pub reason: Option<Arc<str>>,
    /// Smallest row count the governing record measured.
    pub scope_n: Option<u64>,
    /// Record dependence label.
    pub scope_dependence: Option<Arc<str>>,
    /// Commit the governing record was measured at.
    pub calibration_sha: Option<Arc<str>>,
    /// Largest row count the governing record measured.
    pub scope_n_max: Option<u64>,
    /// Nominal level of the governing record.
    pub nominal: Option<f64>,
    /// Coverage the governing record observed.
    pub observed: Option<f64>,
    /// Match key and scope facts the slot was computed from.
    pub basis: Option<CalibrationBasis>,
    /// Slots of further intervals reported beside the primary one.
    pub secondary: Arc<[CalibrationView]>,
}

impl Default for CalibrationView {
    fn default() -> Self {
        Self {
            status: Arc::from("unavailable"),
            record_id: None,
            reason: Some(Arc::from(antecedent_core::reason_code!("not_executed"))),
            scope_n: None,
            scope_dependence: None,
            calibration_sha: None,
            scope_n_max: None,
            nominal: None,
            observed: None,
            basis: None,
            secondary: Arc::from([]),
        }
    }
}

/// Caller-attested evidence (custom validators or an external estimate).
#[derive(Clone, Debug, PartialEq)]
pub struct AttestedEvidence {
    /// Validator name, or the caller-supplied learner label.
    pub name: Arc<str>,
    /// Evidence kind (`custom_validator` or `external_estimate`).
    pub kind: Arc<str>,
    /// Whether the validator passed.
    pub passed: bool,
    /// Refuted ATE, when reported.
    pub refuted_ate: Option<f64>,
    /// Comparison value, when reported.
    pub comparison: Option<f64>,
    /// Whether the result is informative.
    pub informative: bool,
    /// Failure condition, when reported.
    pub failure_condition: Option<Arc<str>>,
    /// Always false — attested evidence is not re-verifiable.
    pub reverifiable: bool,
    /// Digest of the caller-supplied learner config, when this is an external estimate.
    pub config_digest: Option<Arc<str>>,
    /// Digest of the caller-supplied estimate payload, when this is an external estimate.
    pub payload_digest: Option<Arc<str>>,
}

impl ClaimEnvelope {
    /// Construct a claim envelope.
    #[must_use]
    #[allow(
        clippy::too_many_arguments,
        reason = "all-fields constructor of a public record; the arguments are the record own fields, so a parameter struct would only restate it"
    )]
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
            calibration: CalibrationView::default(),
            attested: Arc::from([]),
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
    /// Whether the verified artifact carries a claim over an execution.
    /// A verified program without a claim is not an accepted claim.
    pub claim_present: bool,
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
            claim_present: false,
        }
    }

    /// Record whether the artifact carries a claim.
    #[must_use]
    pub const fn with_claim(mut self, present: bool) -> Self {
        self.claim_present = present;
        self
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
            claim_present: false,
        }
    }

    /// Whether this report accepts the claim as a usable causal claim.
    ///
    /// Requires a verified program that carries a claim.
    #[must_use]
    pub fn accepts_as_claim(&self) -> bool {
        self.accepts_as_verified_program() && self.claim_present
    }

    /// Whether the artifact is a fully verified program, with or without a claim.
    ///
    /// Distinct from storage/forwarding. Old artifacts without a contract
    /// section remain readable but do not pass this check.
    #[must_use]
    pub fn accepts_as_verified_program(&self) -> bool {
        self.recognized && self.verified_references && self.unresolved.is_empty()
    }
}

/// Handoff / loss receipt. Chained receipts accumulate unresolved losses.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct HandoffReceipt {
    /// Input claim id. A derivation records its first parent here; the complete
    /// parent set is [`DerivedClaim::parents`].
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
    #[allow(
        clippy::too_many_arguments,
        reason = "all-fields constructor of a public record; the arguments are the record own fields, so a parameter struct would only restate it"
    )]
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

    /// Whether the output is the same claim with no required meaning omitted.
    ///
    /// The envelope, identities, reasoning and domains must each be explicitly
    /// retained. A derived claim has a new id and is never equivalent to its parent.
    #[must_use]
    pub fn equivalent_claim(&self) -> bool {
        self.omitted.is_empty()
            && self.unresolved.is_empty()
            && self.output == Some(self.input)
            && ["claim.envelope", "identities", "reasoning", "domains"]
                .iter()
                .all(|required| self.retained.iter().any(|field| field.as_ref() == *required))
    }

    /// Accumulate losses from `self` then `next`.
    ///
    /// Omissions, unresolved obligations and unavailable operations only grow
    /// along a chain; a later hop cannot restore what an earlier one lost, so
    /// complete forwarding hops preserve the other hop's retained fields. Between
    /// other hops, only exact field names declared retained by both are established,
    /// excluding everything omitted. Receipts about
    /// different claims (`self.output != Some(next.input)`) do not describe one
    /// path, so the chain retains no established fields, records
    /// `chain_discontinuity` as unresolved, and is never equivalent to the claim.
    #[must_use]
    pub fn chain(&self, next: &Self) -> Self {
        let omitted = merge_unique(&self.omitted, &next.omitted);
        // A complete forwarding hop preserves the other hop's declared fields,
        // including granular fields such as `value` within the full envelope.
        // Otherwise only names explicitly retained at both hops are established.
        let contiguous = self.output == Some(next.input);
        let first_complete = self.equivalent_claim();
        let next_complete = next.equivalent_claim();
        let candidates = if next_complete { &self.retained } else { &next.retained };
        let retained: Arc<[Arc<str>]> = candidates
            .iter()
            .filter(|field| {
                contiguous
                    && (first_complete
                        || next_complete
                        || self.retained.iter().any(|first| first == *field))
                    && !omitted.iter().any(|lost| lost == *field)
            })
            .cloned()
            .collect();
        let mut unresolved = merge_unique(&self.unresolved, &next.unresolved);
        if !contiguous {
            unresolved = merge_unique(&unresolved, &[Arc::from("chain_discontinuity")]);
        }
        Self {
            input: self.input,
            output: next.output,
            consumer: next.consumer.clone(),
            rule: next.rule.clone(),
            retained,
            omitted,
            unresolved,
            unavailable_operations: merge_unique(
                &self.unavailable_operations,
                &next.unavailable_operations,
            ),
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
    /// Pool / synthesize. Unlicensed; not transport.
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
                .find(|item| item.starts_with("alignment:") || alignment_kind(item).is_some())
                .cloned(),
            dependence: EvidenceDependence::Unknown,
        }
    }

    /// Same snapshot is one source. Matching identification is same premises, not same data.
    #[must_use]
    pub fn same_source(&self, other: &Self) -> bool {
        self.snapshot == other.snapshot
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
    /// The claim was valid for the snapshot that produced it. Freshness never
    /// revokes this: withdrawal and supersession preserve the original claim and
    /// only change [`Self::currently_applicable`].
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
        // Withdrawal and supersession are known facts about the claim, not readings
        // of a snapshot: they make it inapplicable even for an offline receiver.
        let currently_applicable = if withdrawn || superseded_by.is_some() {
            Some(false)
        } else {
            current_snapshot.map(|snapshot| snapshot == claim.identities.data_snapshot)
        };
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

/// Alignment kinds a claim may declare for a joint uncertainty contract.
const ALIGNMENT_KINDS: [&str; 3] = ["joint_draws", "covariance", "replicate_ids"];

/// Kind of an alignment evidence string (`kind` or `kind:<id>`, optionally `alignment:`-prefixed).
fn alignment_kind(value: &str) -> Option<&str> {
    let stripped = value.strip_prefix("alignment:").unwrap_or(value);
    let kind = stripped.split_once(':').map_or(stripped, |(kind, _)| kind);
    ALIGNMENT_KINDS.contains(&kind).then_some(kind)
}

/// The shared-draw identity of an alignment declaration: `<kind>:<id>` with a non-empty id.
///
/// A bare kind names a mechanism but not *which* draw set, so it cannot show that
/// two claims share one.
fn alignment_token(value: &str) -> Option<&str> {
    let stripped = value.strip_prefix("alignment:").unwrap_or(value);
    let (kind, id) = stripped.split_once(':')?;
    (ALIGNMENT_KINDS.contains(&kind) && !id.is_empty()).then_some(stripped)
}

/// Alignment is licensed only when every parent declares the same draw set
/// (`kind:id` equality) and was computed on the same data snapshot: replicates
/// from different snapshots are unrelated draws however they are labelled.
fn alignment_is_licensed(parents: &[&ClaimEnvelope], alignment: Option<&str>) -> bool {
    let snapshot = parents.first().map(|parent| parent.identities.data_snapshot);
    if parents.iter().any(|parent| Some(parent.identities.data_snapshot) != snapshot) {
        return false;
    }
    let parent_alignments: Vec<Option<String>> = parents
        .iter()
        .map(|parent| parent.evidence_ref().alignment.map(|item| item.to_string()))
        .collect();
    let parent_tokens: Vec<Option<&str>> =
        parent_alignments.iter().map(|item| item.as_deref().and_then(alignment_token)).collect();
    let Some(expected) = alignment
        .and_then(alignment_token)
        .or_else(|| parent_tokens.iter().copied().flatten().next())
    else {
        return false;
    };
    parent_tokens.iter().all(|item| *item == Some(expected))
}

/// One point-valued parent and the scientific target that its target digest names.
/// The caller must separately establish the authenticity of the parent claim.
#[derive(Clone, Copy, Debug)]
pub struct WorldPointParent<'a> {
    /// Input claim, normally independently accepted or verified.
    pub claim: &'a ClaimEnvelope,
    /// The single-world target whose identity is bound by the claim.
    pub target: &'a ScientificTarget,
}

/// Checked arithmetic over world point estimates. This is not a new claim envelope:
/// it does not establish identification, a program, uncertainty or calibration for
/// the resulting quantity. Its proof identity binds the target, operands and coefficients.
#[derive(Clone, Debug)]
pub struct CheckedWorldPoint {
    target: Arc<ScientificTarget>,
    value: f64,
    parent_ids: Arc<[SemanticDigest]>,
    evidence: Arc<[Arc<str>]>,
    proof_identity: SemanticDigest,
    _retained: Arc<antecedent_core::execution::AdmissionLease>,
}

impl CheckedWorldPoint {
    /// The exact declared linear contrast target.
    #[must_use]
    pub fn target(&self) -> &ScientificTarget {
        &self.target
    }
    /// Weighted sum of the parent point estimates.
    #[must_use]
    pub const fn value(&self) -> f64 {
        self.value
    }
    /// Parent claims ordered by their scientific world identity.
    #[must_use]
    pub fn parent_ids(&self) -> &[SemanticDigest] {
        &self.parent_ids
    }
    /// Union of the evidence references supplied by the parents.
    #[must_use]
    pub fn evidence(&self) -> &[Arc<str>] {
        &self.evidence
    }
    /// Scientific work still required before treating the result as a full claim.
    #[must_use]
    pub const fn limitations(&self) -> &'static [&'static str] {
        &[
            "identification_not_established",
            "uncertainty_not_propagated",
            "calibration_not_transferred",
        ]
    }
    /// Identity of this checked arithmetic, not a claim or contract identity.
    #[must_use]
    pub const fn proof_identity(&self) -> SemanticDigest {
        self.proof_identity
    }
}

/// Caller-selected work, target payload and retained-memory budgets.
/// Admission is acquired before the iterative quote and is held by the result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CheckedWorldPointLimits {
    /// Maximum combined target traversal work across all targets.
    pub max_target_work: usize,
    /// Maximum combined bytes in target payloads.
    pub max_target_payload_bytes: usize,
    /// Maximum action/expression nesting, additionally capped by the encoder's validated depth.
    pub max_target_depth: usize,
    /// Caller-authorized bytes reserved from the execution context throughout the result lifetime.
    pub max_memory_bytes: u64,
    /// Maximum contrast terms / parent world witnesses.
    pub max_terms: usize,
    /// Maximum worst-case term-to-witness comparisons.
    pub max_comparisons: usize,
    /// Maximum total top-level world actions over the target and witnesses.
    pub max_world_actions: usize,
    /// Maximum total assumption records over the target and witnesses.
    pub max_assumption_records: usize,
    /// Maximum evidence references across all parents.
    pub max_evidence_references: usize,
    /// Maximum combined UTF-8 bytes of evidence reference text.
    pub max_evidence_bytes: usize,
}

fn outer_action_count(world: &antecedent_core::InterventionWorld) -> usize {
    let action = world.action();
    action.at_decision_interventions().map_or_else(
        || action.structural_actions().map_or(1, |items| items.len()),
        |items| items.len(),
    )
}

/// Why a checked world-point composition cannot be evaluated.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum CheckedWorldPointError {
    /// The declared target is not a linear world contrast.
    #[error("expected a linear world contrast target")]
    ExpectedContrast,
    /// The parent count does not match the contrast terms.
    #[error("parent count does not match the contrast worlds")]
    ParentCount,
    /// Execution was cancelled.
    #[error("world composition cancelled")]
    Cancelled,
    /// Memory admission was refused.
    #[error("world composition memory admission refused")]
    Admission,
    /// A caller-supplied structural work limit is exceeded.
    #[error("world composition work limit exceeded")]
    TermLimit,
    /// History-indexed laws need a separately admitted verifier.
    #[error("history-indexed target law is not supported by this point verifier")]
    HistoryLawNotSupported,
    /// The same claim identity was supplied twice.
    #[error("duplicate parent claim identity")]
    DuplicateParent,
    /// Parent kind, finite point, observation, data, units or assumption slots differ.
    #[error("incompatible world point parents")]
    IncompatibleParents,
    /// The evidence references exceed the bounded count or text budget.
    #[error("world evidence limit exceeded")]
    EvidenceLimit,
    /// A contrast term names no world in its declared target.
    #[error("contrast world not found")]
    WorldNotFound,
    /// Parent target evidence does not match exactly one term.
    #[error("world target witness mismatch")]
    WitnessMismatch,
    /// An arithmetic contribution or result is not finite.
    #[error("world arithmetic is not finite")]
    NonFiniteArithmetic,
}

/// Reserve a caller-authorized composition workspace and quote each borrowed target.
/// The returned lease must be retained while any output allocation remains live.
/// # Errors
/// Cancellation, admission, arithmetic overflow, target depth or work limits.
pub fn admit_composition_targets<'a>(
    targets: impl IntoIterator<Item = &'a ScientificTarget>,
    premise_quote: antecedent_core::scientific_target::CompositionTargetQuote,
    limits: &CheckedWorldPointLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<Arc<antecedent_core::execution::AdmissionLease>, CheckedWorldPointError> {
    if ctx.cancellation.is_cancelled() {
        return Err(CheckedWorldPointError::Cancelled);
    }
    // The quote's explicit stacks are charged before their first allocation.
    let stack_bytes = limits
        .max_target_work
        .checked_mul(128)
        .and_then(|v| v.checked_add(4096))
        .ok_or(CheckedWorldPointError::TermLimit)?;
    if u64::try_from(stack_bytes).map_err(|_| CheckedWorldPointError::TermLimit)?
        > limits.max_memory_bytes
    {
        return Err(CheckedWorldPointError::TermLimit);
    }
    let lease =
        Arc::new(ctx.acquire_admission(0, limits.max_memory_bytes).map_err(
            |error| match error {
                antecedent_core::execution::AdmissionError::Cancelled => {
                    CheckedWorldPointError::Cancelled
                }
                _ => CheckedWorldPointError::Admission,
            },
        )?);
    let mut work = 0usize;
    let mut payload = premise_quote.payload_bytes;
    for target in targets {
        // Apply the remaining global budgets to this traversal, including the
        // already-quoted parent premises, rather than charging after a full pass.
        let remaining_work = limits
            .max_target_work
            .checked_sub(premise_quote.work)
            .and_then(|v| v.checked_sub(work))
            .ok_or(CheckedWorldPointError::TermLimit)?;
        let remaining_payload = limits
            .max_target_payload_bytes
            .checked_sub(payload)
            .ok_or(CheckedWorldPointError::TermLimit)?;
        let qlimits = antecedent_core::scientific_target::CompositionQuoteLimits {
            max_work: remaining_work,
            max_payload_bytes: remaining_payload,
            max_depth: limits.max_target_depth,
        };
        let q = antecedent_core::scientific_target::quote_composition_target(target, &qlimits, ctx)
            .map_err(|_| {
                if ctx.cancellation.is_cancelled() {
                    CheckedWorldPointError::Cancelled
                } else {
                    CheckedWorldPointError::TermLimit
                }
            })?;
        work = work.checked_add(q.work).ok_or(CheckedWorldPointError::TermLimit)?;
        payload = payload.checked_add(q.payload_bytes).ok_or(CheckedWorldPointError::TermLimit)?;
        if work > limits.max_target_work || payload > limits.max_target_payload_bytes {
            return Err(CheckedWorldPointError::TermLimit);
        }
    }
    // Every target may be encoded repeatedly for each world witness. Charge
    // those passes and comparisons, not only the visitor's first traversal.
    let repetitions = limits.max_terms.checked_add(2).ok_or(CheckedWorldPointError::TermLimit)?;
    let total_work = work
        .checked_mul(repetitions)
        .and_then(|v| v.checked_add(premise_quote.work))
        .and_then(|v| v.checked_add(limits.max_comparisons))
        .and_then(|v| {
            limits
                .max_terms
                .checked_mul(limits.max_terms)
                .and_then(|n| n.checked_mul(4))
                .and_then(|n| v.checked_add(n))
        })
        .ok_or(CheckedWorldPointError::TermLimit)?;
    if total_work > limits.max_target_work || payload > limits.max_target_payload_bytes {
        return Err(CheckedWorldPointError::TermLimit);
    }
    // Covers visitor stacks, worst repeated world/assumption encoding, canonical
    // sorted buffers, result target clone, parent vectors, evidence tree and text.
    let target_bytes = work
        .checked_mul(128)
        .and_then(|v| payload.checked_mul(4).and_then(|p| v.checked_add(p)))
        .and_then(|v| v.checked_mul(repetitions))
        .ok_or(CheckedWorldPointError::TermLimit)?;
    let evidence_bytes = limits
        .max_evidence_references
        .checked_mul(256)
        .and_then(|v| v.checked_add(limits.max_evidence_bytes.checked_mul(2)?))
        .ok_or(CheckedWorldPointError::TermLimit)?;
    let total = target_bytes
        .checked_add(evidence_bytes)
        .and_then(|v| v.checked_add(4096))
        .ok_or(CheckedWorldPointError::TermLimit)?;
    if u64::try_from(total).map_err(|_| CheckedWorldPointError::TermLimit)?
        > limits.max_memory_bytes
    {
        return Err(CheckedWorldPointError::TermLimit);
    }
    Ok(lease)
}

fn quote_parent_premises(
    parents: &[WorldPointParent<'_>],
    limits: &CheckedWorldPointLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_core::scientific_target::CompositionTargetQuote, CheckedWorldPointError> {
    use crate::reasoning::SlotAvailability;
    use antecedent_core::assumption::{Assumption, AssumptionSource};
    let mut work = 0usize;
    let mut charge = |n: usize| -> Result<(), CheckedWorldPointError> {
        work = work.checked_add(n).ok_or(CheckedWorldPointError::TermLimit)?;
        if work > limits.max_target_work {
            return Err(CheckedWorldPointError::TermLimit);
        }
        Ok(())
    };
    for parent in parents {
        if ctx.cancellation.is_cancelled() {
            return Err(CheckedWorldPointError::Cancelled);
        }
        if let Some(units) = &parent.claim.outcome_units {
            charge(units.len())?;
        }
        match &parent.claim.reasoning.assumptions {
            SlotAvailability::Unavailable { reason } => charge(reason.len())?,
            SlotAvailability::Available(slot) => {
                charge(slot.obligations.len())?;
                for obligation in slot.obligations.iter() {
                    if ctx.cancellation.is_cancelled() {
                        return Err(CheckedWorldPointError::Cancelled);
                    }
                    charge(obligation.id.len())?;
                    charge(obligation.description.len())?;
                    if let Some(x) = &obligation.required_check {
                        charge(x.len())?;
                    }
                    if let Some(x) = &obligation.evidence {
                        charge(x.len())?;
                    }
                    match &obligation.source {
                        AssumptionSource::AlgorithmDefault { algorithm } => {
                            charge(algorithm.len())?
                        }
                        AssumptionSource::Derived { from } => charge(from.len())?,
                        _ => {}
                    }
                    match &obligation.assumption {
                        Some(Assumption::Custom { id, description }) => {
                            charge(id.len())?;
                            charge(description.len())?;
                        }
                        Some(Assumption::ParametricRestriction(p)) => {
                            charge(p.id.len())?;
                            charge(p.description.len())?;
                        }
                        Some(Assumption::PriorRestriction(p)) => {
                            charge(p.id.len())?;
                            charge(p.description.len())?;
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    let mut evidence_count = 0usize;
    let mut evidence_bytes = 0usize;
    for parent in parents {
        evidence_count = evidence_count
            .checked_add(parent.claim.evidence.len())
            .ok_or(CheckedWorldPointError::EvidenceLimit)?;
        if evidence_count > limits.max_evidence_references {
            return Err(CheckedWorldPointError::EvidenceLimit);
        }
        for item in parent.claim.evidence.iter() {
            if ctx.cancellation.is_cancelled() {
                return Err(CheckedWorldPointError::Cancelled);
            }
            evidence_bytes = evidence_bytes
                .checked_add(item.len())
                .ok_or(CheckedWorldPointError::EvidenceLimit)?;
            if evidence_bytes > limits.max_evidence_bytes {
                return Err(CheckedWorldPointError::EvidenceLimit);
            }
        }
    }
    // Upper bound repeated equality and lexicographic tree comparisons; no
    // unbounded evidence or obligation text is compared before this check.
    let base = work
        .checked_add(evidence_bytes)
        .and_then(|v| v.checked_add(evidence_count))
        .ok_or(CheckedWorldPointError::TermLimit)?;
    let passes = parents
        .len()
        .checked_add(evidence_count)
        .and_then(|v| v.checked_add(2))
        .ok_or(CheckedWorldPointError::TermLimit)?;
    Ok(antecedent_core::scientific_target::CompositionTargetQuote {
        work: base.checked_mul(passes).ok_or(CheckedWorldPointError::TermLimit)?,
        payload_bytes: base,
    })
}

fn encode_parent_premises(writer: &mut CanonicalWriter, claim: &ClaimEnvelope) {
    use crate::reasoning::SlotAvailability;
    use antecedent_core::CanonicalEncode;
    use antecedent_core::assumption::AssumptionSource;
    writer.text(claim.outcome_units.as_deref().unwrap_or(""));
    match &claim.reasoning.assumptions {
        SlotAvailability::Unavailable { reason } => {
            writer.word(0).text(reason);
        }
        SlotAvailability::Available(slot) => {
            writer.word(1).word(slot.obligations.len() as u64);
            for obligation in slot.obligations.iter() {
                writer.text(&obligation.id).text(obligation.scope.as_str());
                if let crate::obligation::ObligationScope::Horizon { horizon } = obligation.scope {
                    writer.word(u64::from(horizon));
                }
                writer
                    .text(obligation.kind.as_str())
                    .text(obligation.status.as_str())
                    .text(&obligation.description);
                match &obligation.source {
                    AssumptionSource::UserDeclared => {
                        writer.word(1);
                    }
                    AssumptionSource::AlgorithmDefault { algorithm } => {
                        writer.word(2).text(algorithm);
                    }
                    AssumptionSource::Artifact => {
                        writer.word(3);
                    }
                    AssumptionSource::Derived { from } => {
                        writer.word(4).text(from);
                    }
                }
                for text in [&obligation.required_check, &obligation.evidence] {
                    match text {
                        Some(text) => {
                            writer.word(1).text(text);
                        }
                        None => {
                            writer.word(0);
                        }
                    }
                }
                match &obligation.assumption {
                    Some(assumption) => {
                        writer.word(1);
                        assumption.encode(writer);
                    }
                    None => {
                        writer.word(0);
                    }
                }
            }
        }
    }
}

/// Check and evaluate a declared linear combination of point-valued world claims.
///
/// Inputs are paired with contrast terms by their typed world identity, not by the
/// order supplied. The target's assumptions and any history-indexed law must match
/// each witnessed world target. History-indexed target laws are not supported by
/// this bounded verifier. Every parent must be a finite point over the same
/// observation, data snapshot, outcome units and assumptions slot. This operation
/// does not derive intervals or apply a coverage record. To propagate uncertainty,
/// use the typed [`crate::joint_law_rule::check_propagation`] and joint-law
/// propagation APIs with an appropriate dependence declaration. Callers must
/// verify parent claim authenticity separately.
///
/// # Errors
/// Returns a typed error when the target, witness, parent coordinates, reasoning,
/// numerical operands or result fail verification, or the caller's limits are
/// exceeded. See [`CheckedWorldPointLimits`] for caller-selected bounds. Cancellation
/// is checked between bounded stages; individual canonical hashes and sorts do not poll.
pub fn compose_checked_world_points(
    target: &ScientificTarget,
    parents: &[WorldPointParent<'_>],
    limits: &CheckedWorldPointLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<CheckedWorldPoint, CheckedWorldPointError> {
    if ctx.cancellation.is_cancelled() {
        return Err(CheckedWorldPointError::Cancelled);
    }
    if parents.len() > limits.max_terms {
        return Err(CheckedWorldPointError::TermLimit);
    }
    let TargetForm::Contrast(contrast) = target.form() else {
        return Err(CheckedWorldPointError::ExpectedContrast);
    };
    if target.history_indexed_law().is_some() {
        return Err(CheckedWorldPointError::HistoryLawNotSupported);
    }
    let world_count = contrast.worlds().worlds().len();
    if world_count > limits.max_terms
        || world_count.checked_mul(parents.len()).is_none_or(|n| n > limits.max_comparisons)
    {
        return Err(CheckedWorldPointError::TermLimit);
    }
    if world_count < 2 || parents.len() != world_count {
        return Err(CheckedWorldPointError::ParentCount);
    }
    // These are borrowed metadata checks only. Reject unsupported parents before
    // quoting their contents, so the public error identifies the refusal.
    for parent in parents {
        if ctx.cancellation.is_cancelled() {
            return Err(CheckedWorldPointError::Cancelled);
        }
        if parent.target.history_indexed_law().is_some() {
            return Err(CheckedWorldPointError::HistoryLawNotSupported);
        }
        if !matches!(parent.target.form(), TargetForm::World(_)) {
            return Err(CheckedWorldPointError::WitnessMismatch);
        }
    }
    let premise_quote = quote_parent_premises(parents, limits, ctx)?;
    let lease = admit_composition_targets(
        std::iter::once(target).chain(parents.iter().map(|p| p.target)),
        premise_quote,
        limits,
        ctx,
    )?;
    let mut action_count = 0usize;
    let mut assumption_count = target.assumptions().records().len();
    for world in contrast.worlds().worlds() {
        action_count = action_count
            .checked_add(outer_action_count(world))
            .ok_or(CheckedWorldPointError::TermLimit)?;
    }
    for parent in parents {
        if parent.target.history_indexed_law().is_some() {
            return Err(CheckedWorldPointError::HistoryLawNotSupported);
        }
        let TargetForm::World(world) = parent.target.form() else {
            return Err(CheckedWorldPointError::WitnessMismatch);
        };
        action_count = action_count
            .checked_add(outer_action_count(world))
            .ok_or(CheckedWorldPointError::TermLimit)?;
        assumption_count = assumption_count
            .checked_add(parent.target.assumptions().records().len())
            .ok_or(CheckedWorldPointError::TermLimit)?;
    }
    if action_count > limits.max_world_actions || assumption_count > limits.max_assumption_records {
        return Err(CheckedWorldPointError::TermLimit);
    }
    let terms = contrast.terms();
    if terms.len() != world_count {
        return Err(CheckedWorldPointError::ParentCount);
    }
    let first = parents.first().ok_or(CheckedWorldPointError::ParentCount)?.claim;
    let mut distinct_ids = std::collections::BTreeSet::new();
    if parents.iter().any(|p| !distinct_ids.insert(p.claim.claim_id)) {
        return Err(CheckedWorldPointError::DuplicateParent);
    }
    if parents.iter().any(|p| {
        p.claim.kind != ClaimKind::Point
            || p.claim.value.is_none_or(|value| !value.is_finite())
            || p.claim.identities.observation != first.identities.observation
            || p.claim.identities.data_snapshot != first.identities.data_snapshot
            || p.claim.outcome_units != first.outcome_units
            || p.claim.reasoning.assumptions != first.reasoning.assumptions
    }) {
        return Err(CheckedWorldPointError::IncompatibleParents);
    }
    for parent in parents {
        let TargetForm::World(world) = parent.target.form() else {
            return Err(CheckedWorldPointError::WitnessMismatch);
        };
        if parent.claim.outcome_units.as_deref() != Some(world.binding().outcome().units().name()) {
            return Err(CheckedWorldPointError::IncompatibleParents);
        }
    }
    let mut evidence_count = 0usize;
    let mut evidence_bytes = 0usize;
    for parent in parents {
        evidence_count = evidence_count
            .checked_add(parent.claim.evidence.len())
            .ok_or(CheckedWorldPointError::EvidenceLimit)?;
        if evidence_count > limits.max_evidence_references {
            return Err(CheckedWorldPointError::EvidenceLimit);
        }
        for item in parent.claim.evidence.iter() {
            evidence_bytes = evidence_bytes
                .checked_add(item.len())
                .ok_or(CheckedWorldPointError::EvidenceLimit)?;
            if evidence_bytes > limits.max_evidence_bytes {
                return Err(CheckedWorldPointError::EvidenceLimit);
            }
        }
    }
    let witnessed: Vec<_> = parents.iter().map(|p| p.target.identity()).collect();
    let mut ordered_terms = Vec::with_capacity(terms.len());
    for term in terms {
        let world =
            contrast.worlds().get(term.world).ok_or(CheckedWorldPointError::WorldNotFound)?;
        let expected = ScientificTarget::new(
            TargetForm::World(world.clone()),
            target.assumptions().clone(),
            target.inference(),
        )
        .identity();
        ordered_terms.push((expected, term));
    }
    ordered_terms.sort_by_key(|(identity, _)| *identity);
    let mut used = vec![false; parents.len()];
    let mut ids = Vec::with_capacity(ordered_terms.len());
    let mut evidence = std::collections::BTreeSet::new();
    let mut value = 0.0;
    let mut writer = CanonicalWriter::new("antecedent.checked_world_point.v1");
    writer.bytes(target.identity().as_bytes()).word(ordered_terms.len() as u64);
    for (expected, term) in ordered_terms {
        let candidates: Vec<_> = parents
            .iter()
            .enumerate()
            .filter(|(i, p)| {
                !used[*i] && witnessed[*i] == expected && p.claim.identities.target == witnessed[*i]
            })
            .collect();
        if candidates.len() != 1 {
            return Err(CheckedWorldPointError::WitnessMismatch);
        }
        let (index, parent) = candidates[0];
        used[index] = true;
        let operand = parent.claim.value.ok_or(CheckedWorldPointError::IncompatibleParents)?;
        let contribution = term.coefficient * operand;
        if !contribution.is_finite() {
            return Err(CheckedWorldPointError::NonFiniteArithmetic);
        }
        value += contribution;
        if !value.is_finite() {
            return Err(CheckedWorldPointError::NonFiniteArithmetic);
        }
        ids.push(parent.claim.claim_id);
        writer
            .bytes(parent.claim.claim_id.as_bytes())
            .bytes(witnessed[index].as_bytes())
            .real(term.coefficient)
            .real(operand);
        encode_parent_premises(&mut writer, parent.claim);
        let refs = parent.claim.identities.refs();
        writer.word(refs.len() as u64);
        for identity in refs.iter() {
            writer.text(identity.domain.as_str()).bytes(identity.digest.as_bytes());
        }
        writer.word(parent.claim.evidence.len() as u64);
        for item in parent.claim.evidence.iter() {
            writer.text(item);
            evidence.insert(Arc::clone(item));
        }
    }
    if ctx.cancellation.is_cancelled() {
        return Err(CheckedWorldPointError::Cancelled);
    }
    Ok(CheckedWorldPoint {
        target: Arc::new(target.clone()),
        value,
        parent_ids: ids.into(),
        evidence: evidence.into_iter().collect::<Vec<_>>().into(),
        proof_identity: SemanticDigest::from_bytes(writer.finish()),
        _retained: lease,
    })
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
    if let Some(restriction) = derived_departs_from_parents(parents, &derived, operation) {
        return refuse_derived(restriction, parent_ids, Some(compatibility));
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
            operation.as_str(),
            "derive",
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

/// Why `derived` is not a licensed derivation of `parents`, if it is not.
///
/// A retarget stays within its parents' certified derivation: identification
/// premises and product are unchanged. No derivation may drop an
/// identification slot a parent reported, raise identified mass, or claim a
/// domain status a parent did not hold.
fn derived_departs_from_parents(
    parents: &[&ClaimEnvelope],
    derived: &ClaimEnvelope,
    operation: ClaimOperation,
) -> Option<&'static str> {
    if operation == ClaimOperation::Retarget
        && parents.iter().any(|parent| {
            parent.identities.identification != derived.identities.identification
                || parent.identities.identification_product
                    != derived.identities.identification_product
        })
    {
        return Some("retarget_changes_identification");
    }
    // A derivation may weaken a causal claim to a noncausal one but never
    // the reverse: no operation turns an association into a causal statement.
    if parents.iter().any(|parent| parent.kind == ClaimKind::Associational)
        && !matches!(
            derived.kind,
            ClaimKind::Associational | ClaimKind::Refusal | ClaimKind::Incomplete
        )
    {
        return Some("associational_parent_derives_causal_claim");
    }
    if let Some(restriction) = identification_strengthened(parents, derived) {
        return Some(restriction);
    }
    let upgrades =
        [ClaimDomainAxis::Identification, ClaimDomainAxis::Support, ClaimDomainAxis::Evaluated]
            .into_iter()
            .any(|axis| {
                let status = derived.domain(axis);
                status.is_positive() && parents.iter().any(|parent| parent.domain(axis) != status)
            });
    upgrades.then_some("derived_upgrades_domain")
}

/// Identification-slot departures: a dropped slot, malformed masses, more
/// identified mass than a parent, or a stronger status than a parent.
///
/// Comparisons are written so a NaN mass fails them (`!(a <= b)`), never passes.
fn identification_strengthened(
    parents: &[&ClaimEnvelope],
    derived: &ClaimEnvelope,
) -> Option<&'static str> {
    let parent_slots: Vec<_> =
        parents.iter().filter_map(|parent| parent.reasoning.identification.as_ref()).collect();
    let Some(derived_slot) = derived.reasoning.identification.as_ref() else {
        return (!parent_slots.is_empty()).then_some("derived_drops_identification_slot");
    };
    if parent_slots.is_empty() {
        return None;
    }
    if !derived_slot.masses_are_valid() || parent_slots.iter().any(|slot| !slot.masses_are_valid())
    {
        return Some("identification_masses_invalid");
    }
    if parent_slots.iter().any(|slot| {
        // A non-comparable (NaN) mass is a violation, like a larger one.
        derived_slot
            .identified_mass
            .partial_cmp(&(slot.identified_mass + crate::reasoning::IDENTIFICATION_MONOTONE_SLACK))
            .is_none_or(std::cmp::Ordering::is_gt)
    }) {
        return Some("parents_not_retroactively_strengthened");
    }
    let weakest = parent_slots.iter().map(|slot| slot.status.strength_rank()).min();
    if weakest.is_some_and(|rank| derived_slot.status.strength_rank() > rank) {
        return Some("derived_strengthens_identification_status");
    }
    None
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
        assumptions: antecedent_core::assumption::AssumptionSet::new(),
        library_version: Arc::from(antecedent_core::VERSION),
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
    fn same_source_is_snapshot_not_identification() {
        let left = SharedEvidenceRef {
            snapshot: digest(1),
            graph_origin: digest(2),
            prior_origin: None,
            alignment: None,
            dependence: EvidenceDependence::Unknown,
        };
        let same_graph = SharedEvidenceRef { snapshot: digest(3), ..left.clone() };
        let same_snapshot = SharedEvidenceRef { graph_origin: digest(4), ..left.clone() };
        assert!(!left.same_source(&same_graph));
        assert!(left.same_source(&same_snapshot));
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
                assert!(!record.receipt.equivalent_claim());
                assert_eq!(&*record.receipt.rule, "derive");
                assert_eq!(&*record.receipt.consumer, "retarget");
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

    #[test]
    fn retarget_refuses_a_child_outside_the_parent_derivation() {
        use crate::reasoning::SlotAvailability;
        let parent = sample_claim(1, 0.4);
        let refused = |child: ClaimEnvelope| match compose_claims(
            &[&parent],
            ClaimOperation::Retarget,
            Some(child),
            None,
        ) {
            DerivedClaimOutcome::Refused { restriction, parents, .. } => {
                assert_eq!(parents.as_ref(), [parent.claim_id]);
                restriction.to_string()
            }
            other => panic!("expected refusal, got {other:?}"),
        };
        let mut other_product = sample_claim(3, 0.4);
        other_product.identities.identification_product = Some(digest(99));
        assert_eq!(refused(other_product), "retarget_changes_identification");
        let mut other_premises = sample_claim(3, 0.4);
        other_premises.identities.identification = digest(98);
        assert_eq!(refused(other_premises), "retarget_changes_identification");
        let mut dropped = sample_claim(3, 0.4);
        dropped.reasoning.identification = SlotAvailability::unavailable("dropped");
        assert_eq!(refused(dropped), "derived_drops_identification_slot");
        let mut upgraded = sample_claim(3, 0.4);
        upgraded.domains.support = DomainStatus::Supported;
        assert_eq!(refused(upgraded), "derived_upgrades_domain");
        let mut weaker = sample_claim(3, 0.4);
        weaker.domains.evaluated = DomainStatus::Unknown;
        assert!(matches!(
            compose_claims(&[&parent], ClaimOperation::Retarget, Some(weaker), None),
            DerivedClaimOutcome::Derived(_)
        ));
    }

    #[test]
    fn lossless_receipt_alone_is_equivalent() {
        assert!(HandoffReceipt::lossless(digest(1), "host").equivalent_claim());
        let derived =
            HandoffReceipt::new(digest(1), Some(digest(2)), "host", "derive", [], [], [], []);
        assert!(!derived.equivalent_claim());
    }

    #[test]
    fn external_estimate_evidence_is_attested_and_not_reverifiable() {
        let evidence = AttestedEvidence {
            name: Arc::from("econml.dml.CausalForestDML"),
            kind: Arc::from("external_estimate"),
            passed: true,
            refuted_ate: None,
            comparison: None,
            informative: true,
            failure_condition: None,
            reverifiable: false,
            config_digest: Some(Arc::from("aa")),
            payload_digest: Some(Arc::from("bb")),
        };
        assert_eq!(evidence.kind.as_ref(), "external_estimate");
        assert!(!evidence.reverifiable);
        assert_eq!(evidence.config_digest.as_deref(), Some("aa"));
        assert_eq!(evidence.payload_digest.as_deref(), Some("bb"));
    }

    fn with_identification(
        mut claim: ClaimEnvelope,
        status: antecedent_core::identification::IdentificationStatus,
        masses: [f64; 4],
    ) -> ClaimEnvelope {
        use crate::reasoning::{IdentificationSlot, SlotAvailability};
        claim.reasoning.identification = SlotAvailability::Available(IdentificationSlot::new(
            status, masses[0], masses[1], masses[2], masses[3], true, None, false,
        ));
        claim
    }

    fn associational_claim(byte: u8) -> ClaimEnvelope {
        use crate::reasoning::SlotAvailability;
        let mut claim = sample_claim(byte, 1.0);
        claim.kind = ClaimKind::Associational;
        claim.value = None;
        claim.reasoning.identification =
            SlotAvailability::unavailable(crate::reasoning::IDENTIFICATION_NOT_CLAIMED);
        claim.domains = ClaimDomains::new(
            DomainStatus::NotClaimed,
            DomainStatus::Unknown,
            DomainStatus::Evaluated,
        );
        claim
    }

    #[test]
    fn associational_kind_and_domain_have_stable_appended_names() {
        assert_eq!(ClaimKind::Associational.as_str(), "associational");
        assert_eq!(ClaimKind::from_name("associational"), Some(ClaimKind::Associational));
        assert_eq!(ClaimKind::ALL.last(), Some(&ClaimKind::Associational));
        assert_eq!(DomainStatus::NotClaimed.as_str(), "not_claimed");
        assert_eq!(DomainStatus::ALL.last(), Some(&DomainStatus::NotClaimed));
        assert!(!DomainStatus::NotClaimed.is_positive());
        assert!(!ClaimKind::Associational.is_causal());
    }

    #[test]
    fn an_association_cannot_be_retargeted_into_a_causal_claim() {
        let parent = associational_claim(1);
        let mut causal = sample_claim(2, 1.0);
        causal.reasoning.identification = parent.reasoning.identification.clone();
        assert_eq!(
            retarget_refusal(&parent, causal).as_deref(),
            Some("associational_parent_derives_causal_claim")
        );
        // Another noncausal claim over the same association is a licensed derivation.
        assert_eq!(retarget_refusal(&parent, associational_claim(3)), None);
    }

    fn retarget_refusal(parent: &ClaimEnvelope, child: ClaimEnvelope) -> Option<String> {
        match compose_claims(&[parent], ClaimOperation::Retarget, Some(child), None) {
            DerivedClaimOutcome::Refused { restriction, .. } => Some(restriction.to_string()),
            DerivedClaimOutcome::Derived(_) => None,
            other => panic!("unexpected outcome {other:?}"),
        }
    }

    #[test]
    fn withdrawal_is_known_without_a_current_snapshot() {
        let claim = sample_claim(1, 0.4);
        let snapshot = claim.identities.data_snapshot;
        let offline_withdrawn = claim.freshness(Some(snapshot), None, None, true, true);
        assert_eq!(offline_withdrawn.currently_applicable, Some(false));
        let offline_superseded =
            claim.freshness(Some(snapshot), None, Some(digest(99)), false, true);
        assert_eq!(offline_superseded.currently_applicable, Some(false));
        // Not withdrawn and offline: applicability really is unknown.
        assert_eq!(
            claim.freshness(Some(snapshot), None, None, false, true).currently_applicable,
            None
        );
        // Online and matching snapshot: applicable; mismatching: not.
        assert_eq!(
            claim.freshness(Some(snapshot), Some(snapshot), None, false, true).currently_applicable,
            Some(true)
        );
        assert_eq!(
            claim
                .freshness(Some(snapshot), Some(digest(77)), None, false, true)
                .currently_applicable,
            Some(false)
        );
    }

    #[test]
    fn nan_identified_mass_cannot_pass_the_strengthening_gate() {
        use antecedent_core::identification::IdentificationStatus::NonparametricallyIdentified as Np;
        let parent = sample_claim(1, 0.4);
        let nan_child = with_identification(sample_claim(3, 0.4), Np, [f64::NAN, 0.6, 0.0, 0.0]);
        assert_eq!(
            retarget_refusal(&parent, nan_child).as_deref(),
            Some("identification_masses_invalid")
        );
        // A NaN parent mass no longer disables the comparison for that parent.
        let nan_parent = with_identification(sample_claim(1, 0.4), Np, [f64::NAN, 0.6, 0.0, 0.0]);
        assert_eq!(
            retarget_refusal(&nan_parent, sample_claim(3, 0.4)).as_deref(),
            Some("identification_masses_invalid")
        );
        // Masses that are finite but do not partition unit mass are malformed too.
        let inflated = with_identification(sample_claim(3, 0.4), Np, [0.4, 0.6, 0.5, 0.0]);
        assert_eq!(
            retarget_refusal(&parent, inflated).as_deref(),
            Some("identification_masses_invalid")
        );
    }

    #[test]
    fn derived_status_cannot_be_stronger_than_the_weakest_parent() {
        use antecedent_core::identification::IdentificationStatus::{
            NonparametricallyIdentified as Np, NotIdentified as No, PartiallyIdentified as Pi,
        };
        let parent = with_identification(sample_claim(1, 0.0), No, [0.0, 1.0, 0.0, 0.0]);
        let child = with_identification(sample_claim(3, 0.0), Np, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(
            retarget_refusal(&parent, child).as_deref(),
            Some("derived_strengthens_identification_status")
        );
        let weaker = with_identification(sample_claim(3, 0.0), No, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(retarget_refusal(&parent, weaker), None);
        let partial_parent = with_identification(sample_claim(1, 0.4), Pi, [0.4, 0.6, 0.0, 0.0]);
        let np_child = with_identification(sample_claim(3, 0.4), Np, [0.4, 0.6, 0.0, 0.0]);
        assert_eq!(
            retarget_refusal(&partial_parent, np_child).as_deref(),
            Some("derived_strengthens_identification_status")
        );
    }

    #[test]
    fn identification_slot_try_new_requires_a_probability_partition() {
        use crate::reasoning::IdentificationSlot;
        use antecedent_core::identification::IdentificationStatus::NonparametricallyIdentified as Np;
        let build = |m: [f64; 4]| {
            IdentificationSlot::try_new(Np, m[0], m[1], m[2], m[3], true, None, false)
        };
        assert!(build([0.25, 0.25, 0.25, 0.25]).is_ok());
        assert_eq!(build([f64::NAN, 0.0, 0.0, 1.0]).err(), Some("identification_mass_not_finite"));
        assert_eq!(build([1.5, -0.5, 0.0, 0.0]).err(), Some("identification_mass_out_of_range"));
        assert_eq!(build([0.5, 0.2, 0.0, 0.0]).err(), Some("identification_mass_not_partition"));
    }

    fn aligned_claim(byte: u8, snapshot: u8, evidence: &str) -> ClaimEnvelope {
        let mut claim = sample_claim(byte, 0.4);
        claim.identities.data_snapshot = digest(snapshot);
        claim.evidence = Arc::from([Arc::from(evidence)]);
        claim
    }

    #[test]
    fn contrast_alignment_requires_the_same_draw_set_and_snapshot() {
        let contrast = |left: &ClaimEnvelope, right: &ClaimEnvelope| {
            claim_compatibility(&[left, right], ClaimOperation::Contrast, None)
        };
        let a = aligned_claim(1, 16, "alignment:replicate_ids:aa11");
        let same = aligned_claim(2, 16, "alignment:replicate_ids:aa11");
        assert!(contrast(&a, &same).compatible);
        // Same mechanism, different replicate set.
        let other_draws = aligned_claim(2, 16, "alignment:replicate_ids:bb22");
        let report = contrast(&a, &other_draws);
        assert!(!report.compatible);
        assert!(report.unresolved.iter().any(|item| &**item == "alignment"));
        // Same declared draw set id on a different data snapshot is not one draw set.
        let other_snapshot = aligned_claim(2, 17, "alignment:replicate_ids:aa11");
        assert!(!contrast(&a, &other_snapshot).compatible);
        // A bare kind names no draw set at all.
        let bare_a = aligned_claim(1, 16, "replicate_ids");
        let bare_b = aligned_claim(2, 16, "replicate_ids");
        assert!(!contrast(&bare_a, &bare_b).compatible);
        // Different kinds do not align either.
        let joint = aligned_claim(2, 16, "alignment:joint_draws:aa11");
        assert!(!contrast(&a, &joint).compatible);
    }

    #[test]
    fn chained_receipt_keeps_earlier_operation_and_retention_losses() {
        let id = digest(1);
        let chained =
            HandoffReceipt::lossy_scalar(id, "a").chain(&HandoffReceipt::lossless(id, "b"));
        for operation in ["accept_as_claim", "re_estimate"] {
            assert!(
                chained.unavailable_operations.iter().any(|op| &**op == operation),
                "{operation} must stay unavailable"
            );
        }
        for lost in ["reasoning", "domains", "identities"] {
            assert!(chained.omitted.iter().any(|f| &**f == lost));
            assert!(
                !chained.retained.iter().any(|f| &**f == lost),
                "{lost} is omitted and cannot also be retained"
            );
        }
        assert!(!chained.retained.iter().any(|f| &**f == "claim.envelope"));
        assert!(!chained.equivalent_claim());
        assert!(chained.unresolved.is_empty());
    }

    #[test]
    fn incomplete_retention_cannot_be_recovered_by_a_later_hop() {
        let id = digest(1);
        let first =
            HandoffReceipt::new(id, Some(id), "a", "partial", [Arc::from("value")], [], [], []);
        assert!(!first.equivalent_claim());
        let next = HandoffReceipt::new(
            id,
            Some(id),
            "b",
            "forward",
            [Arc::from("value"), Arc::from("claim.envelope")],
            [],
            [],
            [],
        );
        let chained = first.chain(&next);
        assert_eq!(&*chained.retained, &[Arc::<str>::from("value")]);
        assert!(!chained.equivalent_claim());
        assert!(
            HandoffReceipt::lossless(id, "a")
                .chain(&HandoffReceipt::lossless(id, "b"))
                .equivalent_claim()
        );
    }

    #[test]
    fn complete_forwarding_preserves_scalar_retention_in_either_order() {
        let id = digest(1);
        let scalar = HandoffReceipt::lossy_scalar(id, "scalar");
        let complete = HandoffReceipt::lossless(id, "complete");
        for chained in [complete.chain(&scalar), scalar.chain(&complete)] {
            assert_eq!(chained.retained, scalar.retained);
            assert_eq!(chained.omitted, scalar.omitted);
            assert_eq!(chained.unavailable_operations, scalar.unavailable_operations);
            assert!(!chained.equivalent_claim());
        }
        // Naming the same output without naming the retained semantics is not
        // evidence that all four required sections survived.
        let empty = HandoffReceipt::new(id, Some(id), "empty", "unspecified", [], [], [], []);
        assert!(!empty.equivalent_claim());
        assert!(empty.chain(&complete).retained.is_empty());
    }

    #[test]
    fn chaining_receipts_of_different_claims_is_a_discontinuity() {
        let first = HandoffReceipt::lossless(digest(1), "a");
        let second = HandoffReceipt::lossless(digest(2), "b");
        let chained = first.chain(&second);
        assert!(chained.unresolved.iter().any(|item| &**item == "chain_discontinuity"));
        assert!(chained.retained.is_empty());
        assert!(!chained.equivalent_claim());
        for (a, b) in [
            (first.clone(), HandoffReceipt::lossy_scalar(digest(2), "b")),
            (HandoffReceipt::lossy_scalar(digest(1), "a"), second.clone()),
        ] {
            let broken = a.chain(&b);
            assert!(broken.retained.is_empty());
            assert!(broken.unresolved.iter().any(|item| &**item == "chain_discontinuity"));
            assert!(!broken.equivalent_claim());
        }
        // A contiguous chain of lossless hops stays equivalent.
        assert!(first.chain(&HandoffReceipt::lossless(digest(1), "c")).equivalent_claim());
    }

    fn sample_claim(byte: u8, identified_mass: f64) -> ClaimEnvelope {
        use crate::reasoning::{
            AssumptionSlot, IdentificationSlot, ReasoningView, SlotAvailability, SupportSlot,
            UncertaintySlot,
        };
        use antecedent_core::identification::IdentificationStatus;
        ClaimEnvelope::new(
            digest(byte),
            ContractIdentities::new(
                digest(10),
                digest(11),
                Some(digest(12)),
                Some(digest(13)),
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
