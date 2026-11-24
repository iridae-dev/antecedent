//! The declared structure a structural excitation reading is read against.
//!
//! A handwritten label is not evidence for a structural reading. The structure is: one accepted
//! graph or equivalence class, an explicit candidate set, a graph posterior, or a cyclic system,
//! enumerated into members by the shared carrier
//! ([`crate::analysis::structure_members`]) with each member's identification of the
//! exposure's effect on the outcome. A member that orients the exposure into the outcome reads
//! the kernel's mass; a member where the edge is absent or reversed reads zero; an unidentified
//! member reads nothing and its weight is reported beside the reading. Cyclic members are read
//! as lags, since an event process unfolds in time. The reading refuses only when no member
//! identifies it.
//!
//! The bridge evidence is the provenance digest of the search the structure came from, or, for
//! a declared structure, its canonical identity.

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_core::reason_code;
use antecedent_core::reason_details::events;
use serde::Serialize;

use super::EventError;
use crate::analysis::structure_members::{
    CyclicReading, DeclaredStructure, DirectRelation, MemberStatus, StructureForm, StructureMembers,
};

/// How cyclic members of an event process are read: each excitation edge acts after a delay.
pub(super) const CYCLIC: CyclicReading = CyclicReading::Lagged;

/// The members of the declared structure for one exposure and outcome, and the evidence the
/// bridge cites.
pub(super) struct Structural {
    pub(super) members: Arc<StructureMembers>,
    pub(super) evidence: String,
}

/// Which functional the bridge carries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ReadingMode {
    /// The horizon direct mass, by an identity or affine map.
    Direct,
    /// The lifetime total offspring (direct mass and resolvent feedback descendants), by an
    /// identity map only.
    Total,
}

impl ReadingMode {
    pub(super) const fn functional(self) -> &'static str {
        match self {
            Self::Direct => "direct_offspring",
            Self::Total => "total_offspring",
        }
    }

    /// The mode a bridge contract declares; the bridge checks refuse anything else at execution.
    pub(super) fn declared(bridge: &antecedent_contract::bridge::BridgeContract) -> Self {
        if bridge.source.quantity.functional_id == Self::Total.functional() {
            Self::Total
        } else {
            Self::Direct
        }
    }
}

/// Whether the structure is one accepted DAG: its program identities are the ones a single
/// discovered DAG always had.
pub(super) fn single_dag(structure: &DeclaredStructure) -> bool {
    structure.as_graph().is_some_and(|graph| graph.as_dag().is_some())
}

/// Provenance digest of the search the structure came from, as lowercase hex.
pub(super) fn provenance_hex(structure: &DeclaredStructure) -> Option<String> {
    structure.provenance().map(|provenance| crate::accepted::digest_hex(&provenance.digest()))
}

/// Enumerate the members of `structure` for `source` exciting `target`.
///
/// # Errors
/// The structure is not bound to names, the names are not its variables or coincide, a member
/// enumeration refusal, or no member identifies the reading.
pub(super) fn structural_members(
    structure: &DeclaredStructure,
    source: &str,
    target: &str,
    ctx: &ExecutionContext,
) -> Result<Structural, EventError> {
    let names = structure.variable_names().ok_or_else(|| {
        EventError::bridge(
            events::DISCOVERY_UNBOUND,
            reason_code!("schema_mismatch"),
            "the declared structure is not bound to its variable names",
        )
    })?;
    if source == target {
        return Err(EventError::bridge(
            events::SAME_VARIABLE,
            reason_code!("invalid_argument"),
            "a structural reading compares the exposure with a different outcome",
        ));
    }
    if !names.iter().any(|name| name.as_ref() == source)
        || !names.iter().any(|name| name.as_ref() == target)
    {
        return Err(EventError::bridge(
            events::NAMES_NOT_IN_DISCOVERY,
            reason_code!("unknown_variable"),
            "the excitation names are not variables of the declared structure",
        ));
    }
    let members = structure.members(source, target, CYCLIC, ctx)?;
    if !members.any_identified() {
        return Err(EventError::bridge(
            events::READING_NOT_IDENTIFIED,
            reason_code!("effect_not_identified"),
            "no member of the declared structure identifies the exposure's effect on the outcome",
        ));
    }
    let evidence = provenance_hex(structure).unwrap_or_else(|| members.identity().to_hex());
    Ok(Structural { members: Arc::new(members), evidence })
}

/// One member's reading.
#[derive(Clone, Copy, Debug, Serialize)]
#[non_exhaustive]
pub struct MemberReading {
    /// The direct relation between the exposure and the outcome in the member.
    pub relation: DirectRelation,
    /// Normalized weight, when the form has weights.
    pub weight: Option<f64>,
    /// Whether the member identifies the reading.
    pub status: MemberStatus,
    /// The bridged reading of an identified member: the bridged kernel reading where the
    /// member orients the exposure into the outcome, the bridged zero where it does not.
    pub bridged: Option<f64>,
}

/// The structural reading of an accepted excitation over every member of its structure.
#[derive(Clone, Debug, Serialize)]
#[non_exhaustive]
pub struct StructuralReading {
    /// The declared form.
    pub form: StructureForm,
    /// Canonical identity of the declared structure, as lowercase hex.
    pub structure: String,
    /// The evidence the bridge cites: the provenance digest, or the structure's identity.
    pub evidence: String,
    /// The functional the bridge carried: `direct_offspring` or `total_offspring`.
    pub functional: &'static str,
    /// How cyclic members were read.
    pub cyclic: CyclicReading,
    /// One reading per member, in enumeration order.
    pub members: Vec<MemberReading>,
    /// Smallest and largest bridged reading over the identified members.
    pub envelope: [f64; 2],
    /// Mixture conditional on the identified members, for a form with weights.
    pub mixture: Option<f64>,
    /// Weight of the identified members (uniform enumeration weight for a form without
    /// weights).
    pub identified_weight: f64,
    /// A class enumeration stopped before the class was complete.
    pub truncated: bool,
    /// Lifetime resolvent descendants of the outcome, reported beside a carried direct mass
    /// when excitation runs both ways (a two-way kernel or a feedback member). Not a bridged
    /// mean.
    pub cyclic_reading: Option<f64>,
}

/// Mixture weights of the members that read the kernel and of the members that read zero,
/// over the identified members.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Mixture {
    pub(super) oriented: f64,
    pub(super) null: f64,
}

impl Mixture {
    /// Every identified member orients the edge: one law with the added event.
    pub(super) const ORIENTED: Self = Self { oriented: 1.0, null: 0.0 };

    pub(super) fn of(members: &StructureMembers) -> Self {
        let (mut oriented, mut null) = (0.0, 0.0);
        for member in members.members().iter().filter(|member| member.identified()) {
            let weight = members.mixing_weight(member);
            if member.relation().forward() {
                oriented += weight;
            } else {
                null += weight;
            }
        }
        let total = oriented + null;
        if total > 0.0 {
            Self { oriented: oriented / total, null: null / total }
        } else {
            Self::ORIENTED
        }
    }

    /// The groups the law draws: `(added event excites, weight)`, kernel-reading members first.
    pub(super) fn groups(self) -> Vec<(bool, f64)> {
        [(true, self.oriented), (false, self.null)]
            .into_iter()
            .filter(|(_, weight)| *weight > 0.0)
            .collect()
    }
}

/// Read every member: the bridged kernel reading where the member orients the edge, the
/// bridged zero where it does not.
pub(super) fn read(
    structural: &Structural,
    mode: ReadingMode,
    bridged: f64,
    bridged_zero: f64,
    cyclic_reading: Option<f64>,
) -> StructuralReading {
    let members = &structural.members;
    let value = |relation: DirectRelation| if relation.forward() { bridged } else { bridged_zero };
    let summary = members.summarize(|member| value(member.relation()));
    StructuralReading {
        form: members.form(),
        structure: members.identity().to_hex(),
        evidence: structural.evidence.clone(),
        functional: mode.functional(),
        cyclic: members.cyclic_reading(),
        members: members
            .members()
            .iter()
            .map(|member| MemberReading {
                relation: member.relation(),
                weight: member.weight(),
                status: member.status(),
                bridged: member.identified().then(|| value(member.relation())),
            })
            .collect(),
        envelope: summary.envelope.unwrap_or([bridged, bridged]),
        mixture: summary.mixture,
        identified_weight: summary.identified_weight,
        truncated: members.truncated(),
        cyclic_reading,
    }
}

/// Whether any identified member reads zero.
pub(super) fn has_null_member(members: &StructureMembers) -> bool {
    members.members().iter().any(|member| member.identified() && !member.relation().forward())
}

/// Whether any member carries a feedback pair.
pub(super) fn has_feedback_member(members: &StructureMembers) -> bool {
    members.members().iter().any(|member| member.relation().feedback())
}
