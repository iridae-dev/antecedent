//! Migration from the previous release onto the current contracts.
//!
//! A previous-release route record is mapped to the current method records that carry its
//! route: one-to-one, split across several methods, or retired. A closed route keeps its refusal
//! reason as recorded. An explicit error with a remedy is reserved for a genuine break: a licensed
//! route that no current method carries, or one whose licence no current method keeps. A
//! score-row digest is recomputed from wide row ids, and a digest that does not match is refused.
//! A carried scientific target digest is refused unless it is already the current digest.
//! Reduced retention already refuses later operations through
//! `antecedent_state::retention::RetentionMode`. This module does not declare a second retention
//! type. A previous-release artifact is adopted by recomputing the current score-row digest and
//! the current scientific-target digest, then loading the carried receipt. A previous-release
//! route-record artifact is read by [`consume_previous_route_records`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ScientificTarget;

use crate::PayloadDigestWire;
use crate::recalc_receipt_artifact::{RecalcReceiptArtifact, RecalcReceiptArtifactError};

/// Kind string of the score-table row-index digest.
pub const SCORE_ROW_IDENTITY_KIND: &str = "score_reuse.row_index";

/// One migration child: a time frame crossed with an inference realization.
///
/// A geometric cell is not a child.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum RouteAxis {
    /// Static frequentist routes.
    StaticFrequentist,
    /// Static Bayesian routes.
    StaticBayesian,
    /// Temporal frequentist routes.
    TemporalFrequentist,
    /// Temporal Bayesian routes.
    TemporalBayesian,
}

impl RouteAxis {
    /// Every axis, one child each.
    pub const ALL: [Self; 4] = [
        Self::StaticFrequentist,
        Self::StaticBayesian,
        Self::TemporalFrequentist,
        Self::TemporalBayesian,
    ];

    /// Stable name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StaticFrequentist => "static frequentist",
            Self::StaticBayesian => "static bayesian",
            Self::TemporalFrequentist => "temporal frequentist",
            Self::TemporalBayesian => "temporal bayesian",
        }
    }
}

impl std::fmt::Display for RouteAxis {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Status recorded on a public route.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ManifestRouteStatus {
    /// Licensed geometric route.
    Licensed,
    /// Graphless design license.
    Graphless,
    /// Closed geometric route.
    Closed,
    /// Execution without an interval license.
    ExecutionOnly,
    /// Fit or score without a licensed inferential guarantee.
    FittingGap,
}

impl ManifestRouteStatus {
    /// Every status the route manifest records.
    pub const ALL: [Self; 5] =
        [Self::Licensed, Self::Graphless, Self::Closed, Self::ExecutionOnly, Self::FittingGap];
}

/// Scientific-target shape whose identity is content-based.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TargetShape {
    /// One intervention world.
    SingleWorld,
    /// A collection of intervention worlds.
    WorldCollection,
    /// A dose × time grid of those worlds.
    DoseTimeSurface,
    /// A contrast of signed worlds.
    Contrast,
}

/// Why a previous-release value was not migrated.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum MigrationError {
    /// The retained score-row digest is not the digest of the wide row ids.
    #[error(
        "a retained narrow score-row digest is not the current wide-row digest; recompute it \
         from the row ids"
    )]
    ScoreRowDigestNotTranslated,
    /// The carried target digest is not the current scientific-target digest.
    #[error(
        "a carried target digest is not the current scientific-target digest; recompute it from \
         the target"
    )]
    CarriedTargetDigestNotTranslated,
}

/// Current score-table row-index digest.
///
/// Each id is eight little-endian bytes under [`SCORE_ROW_IDENTITY_KIND`].
#[must_use]
pub fn current_score_row_digest(row_ids: &[u64]) -> [u8; 32] {
    PayloadDigestWire::u64s(SCORE_ROW_IDENTITY_KIND, row_ids).digest
}

/// Score-table row-index digest of four-byte ids.
///
/// This is the encoding a retained previous-release digest used. It is not the
/// current digest of a nonempty index. A shared-design batch key still digests
/// a dense four-byte fold position; that key is not this function.
#[must_use]
pub fn narrow_score_row_digest(row_ids: &[u32]) -> [u8; 32] {
    PayloadDigestWire::u32s(SCORE_ROW_IDENTITY_KIND, row_ids).digest
}

/// Keep a retained score-row digest only when it is already the current digest.
///
/// An empty index has no width bytes, so the two encodings coincide. A nonempty
/// four-byte encoding does not. There is no widening of a stored digest.
///
/// # Errors
///
/// `retained` is not [`current_score_row_digest`] of `row_ids`.
#[must_use = "a score-row digest that does not match is refused"]
pub fn adopt_retained_score_row_digest(
    retained: &[u8; 32],
    row_ids: &[u64],
) -> Result<[u8; 32], MigrationError> {
    let current = current_score_row_digest(row_ids);
    if retained == &current {
        Ok(current)
    } else {
        Err(MigrationError::ScoreRowDigestNotTranslated)
    }
}

/// Whether this shape hashes the caller's world id and reference flag.
///
/// None do. A world, a world collection, a dose × time surface, and a contrast
/// are hashed by scientific content. A dose × time surface hashes each cell's
/// content in grid order.
#[must_use]
#[allow(
    clippy::match_same_arms,
    reason = "each shape records that the caller label does not enter"
)]
pub const fn caller_world_id_and_reference_flag_enter_identity(shape: TargetShape) -> bool {
    match shape {
        TargetShape::SingleWorld
        | TargetShape::WorldCollection
        | TargetShape::DoseTimeSurface
        | TargetShape::Contrast => false,
    }
}

/// Keep a carried target digest only when it is already the current digest.
///
/// The previous release has no scientific-target digest to translate. Recompute
/// the digest from the target.
///
/// # Errors
///
/// `retained` is not `current`.
#[must_use = "a target digest that does not match is refused"]
pub fn adopt_carried_target_digest(
    retained: &[u8; 32],
    current: &[u8; 32],
) -> Result<[u8; 32], MigrationError> {
    if retained == current {
        Ok(*current)
    } else {
        Err(MigrationError::CarriedTargetDigestNotTranslated)
    }
}

/// The refusal reason a caller already holds.
///
/// No closed-cell reason was renamed, so this returns `reason` unchanged.
#[must_use]
pub const fn keep_refusal_reason(reason: &str) -> &str {
    reason
}

impl ManifestRouteStatus {
    /// The status named by a route-manifest record, or `None` for an unknown word.
    #[must_use]
    pub fn from_manifest_word(word: &str) -> Option<Self> {
        match word {
            "licensed" => Some(Self::Licensed),
            "graphless" => Some(Self::Graphless),
            "closed" => Some(Self::Closed),
            "execution_only" => Some(Self::ExecutionOnly),
            "fitting_gap" => Some(Self::FittingGap),
            _ => None,
        }
    }
}

/// The fields of one route-manifest record that decide its migration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ManifestRouteRecord<'a> {
    /// Record id.
    pub id: &'a str,
    /// Recorded status word.
    pub status: &'a str,
    /// Recorded design and time model text.
    pub design_time_model: &'a str,
    /// Recorded inference text.
    pub inference: &'a str,
}

/// The axes a record occupies: its time frames crossed with its inference realizations.
///
/// A time frame is named by the words `static` or `temporal` in the design and time model,
/// and a realization by `frequentist` or `bayesian` in the inference text (case ignored).
/// A record that names no time frame or no realization occupies no axis.
#[must_use]
pub fn route_record_axes(design_time_model: &str, inference: &str) -> Vec<RouteAxis> {
    let design = design_time_model.to_ascii_lowercase();
    let inference = inference.to_ascii_lowercase();
    let frames = [("static", false), ("temporal", true)]
        .into_iter()
        .filter(|(word, _)| design.contains(word))
        .map(|(_, temporal)| temporal);
    let realizations = [("frequentist", false), ("bayesian", true)]
        .into_iter()
        .filter(|(word, _)| inference.contains(word))
        .map(|(_, bayesian)| bayesian)
        .collect::<Vec<_>>();
    let mut axes = Vec::new();
    for temporal in frames {
        for bayesian in &realizations {
            axes.push(match (temporal, *bayesian) {
                (false, false) => RouteAxis::StaticFrequentist,
                (false, true) => RouteAxis::StaticBayesian,
                (true, false) => RouteAxis::TemporalFrequentist,
                (true, true) => RouteAxis::TemporalBayesian,
            });
        }
    }
    axes
}

/// One method record of the current route manifest, as far as the migration reads it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CurrentMethod {
    /// Method record id.
    pub id: String,
    /// The previous-release route id this method carries, or `unlisted` for a method that has no
    /// previous-release route.
    pub route: String,
    /// Estimator wire id, or `unlisted`.
    pub estimator: String,
    /// Time frame: `static` or `temporal`.
    pub design: String,
    /// Inference realization: `frequentist`, `bayesian`, `both`, or `unlisted`.
    pub inference: String,
    /// Recorded status word.
    pub status: String,
}

impl CurrentMethod {
    /// The axes this method occupies. A method that records no realization occupies none.
    #[must_use]
    pub fn axes(&self) -> Vec<RouteAxis> {
        let inference =
            if self.inference == "both" { "frequentist bayesian" } else { &self.inference };
        route_record_axes(&self.design, inference)
    }
}

/// One current method that carries a previous-release route.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MethodAssignment {
    /// Current method record id.
    pub method: String,
    /// Estimator wire id of that method.
    pub estimator: String,
    /// Status word of that method.
    pub status: String,
    /// Axes that method occupies.
    pub axes: Vec<RouteAxis>,
    /// The previous route was licensed and this method is not.
    pub narrowed: bool,
}

/// What carrying one route-manifest record onto the current contracts does.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RouteRecordOutcome {
    /// A closed route keeps its refusal reason as recorded. Nothing is rewritten.
    RefusalKept,
    /// One current method carries the route.
    OneToOne {
        /// The method.
        method: MethodAssignment,
        /// Axes the previous record names that the method does not occupy. The method record is
        /// the authority for its time frame and realization.
        unserved_axes: Vec<RouteAxis>,
    },
    /// Several current methods carry the route, one per estimator, structure family or
    /// realization. Each is a separate method with its own evidence and uncertainty.
    Split {
        /// The methods, in inventory order.
        methods: Vec<MethodAssignment>,
        /// Axes the previous record names that no method occupies.
        unserved_axes: Vec<RouteAxis>,
    },
    /// No current method carries the route and it was not licensed. The route left the
    /// inventory; a caller must stop requesting it.
    Retired,
}

/// Why a route-manifest record was not carried onto the current contracts.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum RouteRecordError {
    /// The status word is not one the manifest vocabulary records.
    #[error("route {id} has the unknown status {status}")]
    UnknownStatus {
        /// Record id.
        id: String,
        /// The status word.
        status: String,
    },
    /// A licensed route has no current method.
    #[error(
        "licensed route {id} has no current method; request the query through the method that \
         replaced it, or record the route as retired in the current inventory before migrating"
    )]
    RouteWithoutCurrentMethod {
        /// Record id.
        id: String,
    },
    /// A licensed route is carried only by methods that are not licensed.
    #[error(
        "licensed route {id} is carried only by unlicensed methods {}; run the measurement that \
         licenses one of them, or accept the execution-only standing before migrating",
        .methods.join(", ")
    )]
    LicenseNotCarried {
        /// Record id.
        id: String,
        /// The current methods, none licensed.
        methods: Vec<String>,
    },
}

/// Carry one route-manifest record onto the current method records.
///
/// A closed route that no method carries keeps its recorded refusal ([`keep_refusal_reason`]).
/// Every other record is mapped to the methods whose `route` is its id: one is a one-to-one
/// outcome and several are a split. A non-closed route with no method is retired unless it was
/// licensed, which is a break.
///
/// # Errors
/// The status is unknown, a licensed route has no method, or no method keeps its licence.
#[must_use = "a route record that is not carried onto the current methods is refused"]
pub fn migrate_route_record(
    record: &ManifestRouteRecord<'_>,
    current: &[CurrentMethod],
) -> Result<RouteRecordOutcome, RouteRecordError> {
    let id = record.id.to_owned();
    let status = ManifestRouteStatus::from_manifest_word(record.status).ok_or_else(|| {
        RouteRecordError::UnknownStatus { id: id.clone(), status: record.status.to_owned() }
    })?;
    let linked: Vec<&CurrentMethod> =
        current.iter().filter(|m| m.route != "unlisted" && m.route == record.id).collect();
    if linked.is_empty() {
        return match status {
            ManifestRouteStatus::Closed => Ok(RouteRecordOutcome::RefusalKept),
            ManifestRouteStatus::Licensed => {
                Err(RouteRecordError::RouteWithoutCurrentMethod { id })
            }
            _ => Ok(RouteRecordOutcome::Retired),
        };
    }
    let licensed = status == ManifestRouteStatus::Licensed;
    if licensed && linked.iter().all(|m| m.status != "licensed") {
        return Err(RouteRecordError::LicenseNotCarried {
            id,
            methods: linked.iter().map(|m| m.id.clone()).collect(),
        });
    }
    let mut methods: Vec<MethodAssignment> = linked
        .iter()
        .map(|m| MethodAssignment {
            method: m.id.clone(),
            estimator: m.estimator.clone(),
            status: m.status.clone(),
            axes: m.axes(),
            narrowed: licensed && m.status != "licensed",
        })
        .collect();
    let unserved_axes: Vec<RouteAxis> =
        route_record_axes(record.design_time_model, record.inference)
            .into_iter()
            .filter(|axis| !methods.iter().any(|m| m.axes.contains(axis)))
            .collect();
    if methods.len() == 1 {
        let method = methods.remove(0);
        Ok(RouteRecordOutcome::OneToOne { method, unserved_axes })
    } else {
        Ok(RouteRecordOutcome::Split { methods, unserved_axes })
    }
}

/// Current methods that carry no previous-release route: they are additions, not migrations.
#[must_use]
pub fn added_methods<'a>(
    current: &'a [CurrentMethod],
    previous_route_ids: &[&str],
) -> Vec<&'a CurrentMethod> {
    current
        .iter()
        .filter(|m| m.route == "unlisted" || !previous_route_ids.contains(&m.route.as_str()))
        .collect()
}

/// One intentional break between the previous release and the current contracts.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum IntentionalBreak {
    /// Score-table row ids are eight bytes, not four.
    ScoreRowIdWidth,
    /// The scientific-target digest is the canonical payload at the current payload format.
    ScientificTargetDigest,
    /// World, collection, surface and contrast identities are content-based.
    ContentBasedWorldIdentity,
    /// A reduced retention mode refuses the later operations it no longer serves.
    ReducedRetentionOperation,
    /// A licensed route has no current method.
    RouteWithoutCurrentMethod,
    /// A licensed route is carried only by unlicensed methods.
    LicenseNotCarried,
    /// A closed route keeps its refusal reason.
    RefusalReason,
}

impl IntentionalBreak {
    /// Every break the inventory records.
    pub const ALL: [Self; 7] = [
        Self::ScoreRowIdWidth,
        Self::ScientificTargetDigest,
        Self::ContentBasedWorldIdentity,
        Self::ReducedRetentionOperation,
        Self::RouteWithoutCurrentMethod,
        Self::LicenseNotCarried,
        Self::RefusalReason,
    ];
}

/// The explicit path a caller takes across one break.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BreakPath {
    /// A retained value is not translated. Recompute it from the source, and the retained
    /// value is refused with `error` until then.
    RecomputeFromSource {
        /// The error a retained value gets.
        error: &'static str,
        /// What to recompute it from.
        source: &'static str,
    },
    /// The change is refused, and the refusal names the attempt that reaches the same end.
    Refused {
        /// The error or refusal text.
        error: &'static str,
        /// The attempt that reaches the same end.
        next_attempt: &'static str,
    },
    /// Nothing changed. The previous value is the current value.
    Unchanged {
        /// Why no path is needed.
        reason: &'static str,
    },
}

/// The migration or error path of one break.
#[must_use]
pub const fn break_path(change: IntentionalBreak) -> BreakPath {
    match change {
        IntentionalBreak::ScoreRowIdWidth => BreakPath::RecomputeFromSource {
            error: "MigrationError::ScoreRowDigestNotTranslated",
            source: "the eight-byte row ids",
        },
        IntentionalBreak::ScientificTargetDigest => BreakPath::RecomputeFromSource {
            error: "MigrationError::CarriedTargetDigestNotTranslated",
            source: "the scientific target",
        },
        IntentionalBreak::ContentBasedWorldIdentity => BreakPath::RecomputeFromSource {
            error: "MigrationError::CarriedTargetDigestNotTranslated",
            source: "the scientific target; world ids and the reference flag no longer enter",
        },
        IntentionalBreak::ReducedRetentionOperation => BreakPath::Refused {
            error: "antecedent_state::retention::RetentionRefusal",
            next_attempt: "reload the rows or the disk artifact and run the operation on the \
                           resident state",
        },
        IntentionalBreak::RouteWithoutCurrentMethod => BreakPath::Refused {
            error: "RouteRecordError::RouteWithoutCurrentMethod",
            next_attempt: "request the query through the method that replaced the route",
        },
        IntentionalBreak::LicenseNotCarried => BreakPath::Refused {
            error: "RouteRecordError::LicenseNotCarried",
            next_attempt: "license one of the carrying methods, or accept its execution-only \
                           standing",
        },
        IntentionalBreak::RefusalReason => {
            BreakPath::Unchanged { reason: "no closed-cell reason was renamed" }
        }
    }
}

/// Why a previous-release artifact was not adopted.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum PreviousReleaseError {
    /// A score-row digest or a carried target digest was not the current digest.
    #[error(transparent)]
    Migration(#[from] MigrationError),
    /// A fresh process reused a derived stage. That reuse stays in-process.
    #[error("a fresh process cannot reuse derived stage {stage}; that reuse stays in-process only")]
    FreshProcessReusedDerivedStage {
        /// Stage the receipt marked reused.
        stage: String,
    },
    /// The carried receipt failed a check other than fresh-process reuse.
    #[error("the previous-release receipt was not loaded: {detail}")]
    ReceiptNotLoaded {
        /// The receipt error.
        detail: String,
    },
    /// The artifact bytes are not the previous-release object.
    #[error("the previous-release artifact does not decode")]
    ArtifactNotDecoded,
}

/// What a previous-release artifact adopted under the current contracts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdoptedRelease {
    /// Current eight-byte score-row digest.
    pub score_row_digest: [u8; 32],
    /// Current scientific-target digest.
    pub target_digest: [u8; 32],
    /// `in_process` or `fresh_process`, as the receipt declares it.
    pub boundary_name: String,
    /// Stage label and status, in receipt order.
    pub stage_status: Vec<(String, String)>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PreviousReleaseWire {
    row_ids: Vec<u64>,
    retained_score_digest: [u8; 32],
    carried_target_digest: [u8; 32],
    receipt: Vec<u8>,
}

/// Load a previous-release artifact: wide row ids, a carried target digest, and a receipt.
///
/// A nonempty four-byte score-row digest is refused. A carried target digest is
/// kept only when it is already [`ScientificTarget::identity`]. A fresh-process
/// receipt that reuses a derived stage is refused before the receipt identity is
/// compared. Route axes are not migrated.
///
/// # Errors
/// The bytes do not decode, either digest is not the current digest, or the
/// receipt refuses, including a derived stage reused in a fresh process.
#[must_use = "a previous-release artifact is refused when its digest or receipt does not adopt"]
pub fn consume_previous_release_artifact(
    bytes: &[u8],
    target: &ScientificTarget,
) -> Result<AdoptedRelease, PreviousReleaseError> {
    let wire: PreviousReleaseWire =
        serde_json::from_slice(bytes).map_err(|_| PreviousReleaseError::ArtifactNotDecoded)?;
    let score_row_digest =
        adopt_retained_score_row_digest(&wire.retained_score_digest, &wire.row_ids)?;
    let target_digest =
        adopt_carried_target_digest(&wire.carried_target_digest, target.identity().as_bytes())?;
    let artifact = match RecalcReceiptArtifact::from_bytes(&wire.receipt, None) {
        Ok(artifact) => artifact,
        Err(RecalcReceiptArtifactError::FreshProcessReuse { stage }) => {
            return Err(PreviousReleaseError::FreshProcessReusedDerivedStage { stage });
        }
        Err(error) => {
            return Err(PreviousReleaseError::ReceiptNotLoaded { detail: error.to_string() });
        }
    };
    let meta = artifact.meta();
    Ok(AdoptedRelease {
        score_row_digest,
        target_digest,
        boundary_name: meta.capabilities.boundary.clone(),
        stage_status: meta
            .entries
            .iter()
            .map(|entry| (entry.stage.clone(), entry.status.clone()))
            .collect(),
    })
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PreviousRouteWire {
    id: String,
    status: String,
    design_time_model: String,
    inference: String,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PreviousRoutesWire {
    routes: Vec<PreviousRouteWire>,
}

/// One previous-release route record and what carrying it onto the current methods did.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RouteRecordMigration {
    /// Previous-release route id.
    pub id: String,
    /// The outcome, or the break with its remedy.
    pub result: Result<RouteRecordOutcome, RouteRecordError>,
}

/// Load a previous-release route-record artifact and carry every record onto `current`.
///
/// The artifact is `{"routes":[{"id","status","design_time_model","inference"}, ...]}`. Each
/// record gets its own outcome or error, in artifact order; one break does not hide the others.
///
/// # Errors
/// The bytes do not decode as that object.
#[must_use = "route records are carried or refused individually"]
pub fn consume_previous_route_records(
    bytes: &[u8],
    current: &[CurrentMethod],
) -> Result<Vec<RouteRecordMigration>, PreviousReleaseError> {
    let wire: PreviousRoutesWire =
        serde_json::from_slice(bytes).map_err(|_| PreviousReleaseError::ArtifactNotDecoded)?;
    let mut ids = std::collections::BTreeSet::new();
    if wire.routes.iter().any(|route| {
        route.id.is_empty() || route.id == "unlisted" || !ids.insert(route.id.as_str())
    }) {
        return Err(PreviousReleaseError::ArtifactNotDecoded);
    }
    Ok(wire
        .routes
        .into_iter()
        .map(|route| {
            let result = migrate_route_record(
                &ManifestRouteRecord {
                    id: &route.id,
                    status: &route.status,
                    design_time_model: &route.design_time_model,
                    inference: &route.inference,
                },
                current,
            );
            RouteRecordMigration { id: route.id, result }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use antecedent_core::{
        AssumptionSet, Conditioning, GraphFamily, InferenceSelection, Intervention,
        InterventionContrast, InterventionWorld, ModelAssumptions, ObservationUnit,
        OutcomeQuantity, QuantityUnits, ScientificTarget, ScientificWorldId, TargetForm,
        TargetPopulation, TimeFrame, Value, VariableId, WorldAction, WorldBinding, WorldCollection,
    };

    use super::{
        MigrationError, RouteAxis, TargetShape, adopt_carried_target_digest,
        adopt_retained_score_row_digest, caller_world_id_and_reference_flag_enter_identity,
        current_score_row_digest, keep_refusal_reason, narrow_score_row_digest,
    };

    fn world(id: u32, reference: bool, level: f64) -> InterventionWorld {
        let time = TimeFrame::static_decision();
        let binding = WorldBinding::new(
            TargetPopulation::AllObserved,
            ObservationUnit::new("subject").unwrap(),
            OutcomeQuantity::mean(VariableId::from_raw(1), QuantityUnits::dimensionless()).unwrap(),
            Conditioning::unconditional(),
        )
        .unwrap();
        InterventionWorld::new(
            ScientificWorldId::from_raw(id),
            WorldAction::at_decision([Intervention::set(
                VariableId::from_raw(0),
                Value::f64(level),
            )])
            .unwrap(),
            reference,
            binding,
            time,
        )
        .unwrap()
    }

    fn target(form: TargetForm) -> ScientificTarget {
        ScientificTarget::new(
            form,
            ModelAssumptions::new(GraphFamily::Dag, AssumptionSet::new()),
            InferenceSelection::Unspecified,
        )
    }

    fn contrast(reference_id: u32, active_id: u32) -> ScientificTarget {
        let worlds =
            WorldCollection::new([world(reference_id, true, 0.0), world(active_id, false, 1.0)])
                .unwrap();
        let contrast = InterventionContrast::difference(
            worlds,
            ScientificWorldId::from_raw(active_id),
            ScientificWorldId::from_raw(reference_id),
            ScientificWorldId::from_raw(reference_id),
        )
        .unwrap();
        target(TargetForm::Contrast(contrast))
    }

    #[test]
    fn nonempty_narrow_score_row_digest_is_not_adopted() {
        let row_ids = [1_u64, 2, 3];
        let narrow = narrow_score_row_digest(&[1, 2, 3]);
        let current = current_score_row_digest(&row_ids);
        assert_ne!(narrow, current);
        let error = adopt_retained_score_row_digest(&narrow, &row_ids).unwrap_err();
        assert_eq!(error, MigrationError::ScoreRowDigestNotTranslated);
        assert_eq!(adopt_retained_score_row_digest(&current, &row_ids).unwrap(), current);
    }

    #[test]
    fn empty_score_row_digest_matches_because_the_payload_is_empty() {
        let narrow = narrow_score_row_digest(&[]);
        let current = current_score_row_digest(&[]);
        assert_eq!(narrow, current);
        assert_eq!(adopt_retained_score_row_digest(&narrow, &[]).unwrap(), current);
    }

    #[test]
    fn world_identity_omits_the_caller_label() {
        let labelled = target(TargetForm::World(world(1, true, 0.0)));
        let relabelled = target(TargetForm::World(world(2, true, 0.0)));
        let unflagged = target(TargetForm::World(world(1, false, 0.0)));
        assert_ne!(labelled, relabelled);
        assert_ne!(labelled, unflagged);
        assert_eq!(labelled.identity(), relabelled.identity());
        assert_eq!(labelled.identity(), unflagged.identity());
        assert!(!caller_world_id_and_reference_flag_enter_identity(TargetShape::SingleWorld));
        assert_eq!(
            caller_world_id_and_reference_flag_enter_identity(TargetShape::SingleWorld),
            labelled.identity() != relabelled.identity()
        );

        let collection = target(TargetForm::Worlds(
            WorldCollection::new([world(1, true, 0.0), world(3, false, 1.0)]).unwrap(),
        ));
        let relabelled_collection = target(TargetForm::Worlds(
            WorldCollection::new([world(2, true, 0.0), world(3, false, 1.0)]).unwrap(),
        ));
        assert_ne!(collection, relabelled_collection);
        assert_eq!(collection.identity(), relabelled_collection.identity());
        assert_eq!(
            caller_world_id_and_reference_flag_enter_identity(TargetShape::WorldCollection),
            collection.identity() != relabelled_collection.identity()
        );
        assert!(!caller_world_id_and_reference_flag_enter_identity(TargetShape::DoseTimeSurface));
    }

    #[test]
    fn contrast_identity_omits_the_caller_label() {
        let original = contrast(0, 1);
        let relabelled = contrast(40, 9);
        assert_eq!(original.identity(), relabelled.identity());
        assert!(!caller_world_id_and_reference_flag_enter_identity(TargetShape::Contrast));
        assert_eq!(
            caller_world_id_and_reference_flag_enter_identity(TargetShape::Contrast),
            original.identity() != relabelled.identity()
        );
    }

    #[test]
    fn a_different_target_digest_is_not_adopted() {
        let current = target(TargetForm::World(world(1, true, 0.0))).identity();
        let other = target(TargetForm::World(world(1, true, 1.0))).identity();
        let current_bytes = *current.as_bytes();
        let other_bytes = *other.as_bytes();
        assert_eq!(
            adopt_carried_target_digest(&current_bytes, &current_bytes).unwrap(),
            current_bytes
        );
        let error = adopt_carried_target_digest(&other_bytes, &current_bytes).unwrap_err();
        assert_eq!(error, MigrationError::CarriedTargetDigestNotTranslated);
    }

    #[test]
    fn inventory_refusal_reasons_are_not_rewritten() {
        let reasons = [
            "policy_route_not_licensed",
            "Attribution is licensed only for an explicit Dag at validation none.",
            "AnomalyAttribution and ChangeAttribution require a supplied static Dag. \
             The attribution engine has no lagged latent slice.",
        ];
        for reason in reasons {
            assert_eq!(keep_refusal_reason(reason), reason);
        }
    }

    #[test]
    fn route_records_occupy_the_axes_they_name() {
        use super::route_record_axes;
        assert_eq!(
            route_record_axes("static; structure=explicit", "Frequentist,Bayesian"),
            [RouteAxis::StaticFrequentist, RouteAxis::StaticBayesian]
        );
        assert_eq!(
            route_record_axes("temporal; structure=explicit", "bayesian"),
            [RouteAxis::TemporalBayesian]
        );
        assert!(route_record_axes("binary_aipw", "unlisted").is_empty());
        assert!(route_record_axes("static", "unlisted").is_empty());
    }

    fn method(
        id: &str,
        route: &str,
        design: &str,
        inference: &str,
        status: &str,
    ) -> super::CurrentMethod {
        super::CurrentMethod {
            id: id.into(),
            route: route.into(),
            estimator: "est".into(),
            design: design.into(),
            inference: inference.into(),
            status: status.into(),
        }
    }

    #[test]
    fn a_route_record_maps_one_to_one_split_retired_or_kept() {
        use super::{
            ManifestRouteRecord, RouteRecordError, RouteRecordOutcome, migrate_route_record,
        };
        let record = |id, status, design, inference| ManifestRouteRecord {
            id,
            status,
            design_time_model: design,
            inference,
        };
        let current = [
            method("m.one", "r.one", "static", "frequentist", "licensed"),
            method("m.a", "r.split", "static", "frequentist", "licensed"),
            method("m.b", "r.split", "static", "bayesian", "execution_only"),
            method("m.both", "r.both", "temporal", "both", "licensed"),
            method("m.down", "r.down", "static", "bayesian", "execution_only"),
        ];
        let one = migrate_route_record(
            &record("r.one", "licensed", "static; structure=explicit", "Frequentist"),
            &current,
        )
        .unwrap();
        let RouteRecordOutcome::OneToOne { method, unserved_axes } = one else {
            panic!("expected one-to-one");
        };
        assert_eq!(method.method, "m.one");
        assert_eq!(method.axes, [RouteAxis::StaticFrequentist]);
        assert!(!method.narrowed && unserved_axes.is_empty());

        let split = migrate_route_record(
            &record("r.split", "licensed", "static", "Frequentist,Bayesian"),
            &current,
        )
        .unwrap();
        let RouteRecordOutcome::Split { methods, unserved_axes } = split else {
            panic!("expected a split");
        };
        assert_eq!(methods.iter().map(|m| m.method.as_str()).collect::<Vec<_>>(), ["m.a", "m.b"]);
        assert_eq!(methods.iter().map(|m| m.narrowed).collect::<Vec<_>>(), [false, true]);
        assert!(unserved_axes.is_empty());

        // The method record is the authority on the time frame; the old text over-named one.
        let both = migrate_route_record(
            &record("r.both", "licensed", "static and temporal", "Frequentist"),
            &current,
        )
        .unwrap();
        let RouteRecordOutcome::OneToOne { method, unserved_axes } = both else {
            panic!("expected one-to-one");
        };
        assert_eq!(method.axes, [RouteAxis::TemporalFrequentist, RouteAxis::TemporalBayesian]);
        assert_eq!(unserved_axes, [RouteAxis::StaticFrequentist]);

        assert_eq!(
            migrate_route_record(&record("c", "closed", "static", "refused:Frequentist"), &current),
            Ok(RouteRecordOutcome::RefusalKept)
        );
        assert_eq!(
            migrate_route_record(&record("gone", "execution_only", "static", "x"), &current),
            Ok(RouteRecordOutcome::Retired)
        );
        assert!(matches!(
            migrate_route_record(&record("r", "retired", "static", "bayesian"), &current),
            Err(RouteRecordError::UnknownStatus { .. })
        ));
    }

    #[test]
    fn a_licensed_route_without_a_licensed_method_is_a_break_with_a_remedy() {
        use super::{ManifestRouteRecord, RouteRecordError, migrate_route_record};
        let record = |id| ManifestRouteRecord {
            id,
            status: "licensed",
            design_time_model: "static",
            inference: "bayesian",
        };
        let current = [method("m.down", "r.down", "static", "bayesian", "execution_only")];
        let missing = migrate_route_record(&record("r.gone"), &current).unwrap_err();
        assert_eq!(missing, RouteRecordError::RouteWithoutCurrentMethod { id: "r.gone".into() });
        assert!(missing.to_string().contains("method that replaced it"));
        let downgraded = migrate_route_record(&record("r.down"), &current).unwrap_err();
        assert_eq!(
            downgraded,
            RouteRecordError::LicenseNotCarried {
                id: "r.down".into(),
                methods: vec!["m.down".into()]
            }
        );
        assert!(downgraded.to_string().contains("m.down"));
    }

    #[test]
    fn methods_without_a_previous_route_are_additions() {
        let current = [
            method("m.new", "unlisted", "static", "unlisted", "execution_only"),
            method("m.kept", "r.one", "static", "bayesian", "licensed"),
        ];
        let added = super::added_methods(&current, &["r.one"]);
        assert_eq!(added.len(), 1);
        assert_eq!(added[0].id, "m.new");
        assert!(added[0].axes().is_empty());
    }

    #[test]
    fn reserved_route_never_links_and_invalid_previous_ids_are_refused() {
        use super::{ManifestRouteRecord, PreviousReleaseError, RouteRecordError};
        let current = [method("m.new", "unlisted", "static", "frequentist", "licensed")];
        let record = ManifestRouteRecord {
            id: "unlisted",
            status: "licensed",
            design_time_model: "static",
            inference: "frequentist",
        };
        assert!(matches!(
            super::migrate_route_record(&record, &current),
            Err(RouteRecordError::RouteWithoutCurrentMethod { .. })
        ));
        assert_eq!(super::added_methods(&current, &["unlisted"]).len(), 1);
        for bytes in [
            br#"{"routes":[{"id":"","status":"licensed","design_time_model":"static","inference":"frequentist"}]}"#.as_slice(),
            br#"{"routes":[{"id":"unlisted","status":"licensed","design_time_model":"static","inference":"frequentist"}]}"#.as_slice(),
            br#"{"routes":[{"id":"r","status":"licensed","design_time_model":"static","inference":"frequentist"},{"id":"r","status":"licensed","design_time_model":"static","inference":"frequentist"}]}"#.as_slice(),
        ] {
            assert!(matches!(super::consume_previous_route_records(bytes, &current),
                Err(PreviousReleaseError::ArtifactNotDecoded)));
        }
    }

    #[test]
    fn every_break_has_a_path_and_each_path_matches_the_function_that_enforces_it() {
        use super::{
            BreakPath, IntentionalBreak, ManifestRouteRecord, RouteRecordError, break_path,
            migrate_route_record,
        };
        let licensed = ManifestRouteRecord {
            id: "r",
            status: "licensed",
            design_time_model: "static",
            inference: "bayesian",
        };
        for change in IntentionalBreak::ALL {
            match (change, break_path(change)) {
                (
                    IntentionalBreak::ScoreRowIdWidth,
                    BreakPath::RecomputeFromSource { error, .. },
                ) => {
                    let narrow = narrow_score_row_digest(&[1, 2]);
                    let refused = adopt_retained_score_row_digest(&narrow, &[1, 2]).unwrap_err();
                    assert!(error.ends_with("ScoreRowDigestNotTranslated"));
                    assert_eq!(refused, MigrationError::ScoreRowDigestNotTranslated);
                }
                (
                    IntentionalBreak::ScientificTargetDigest
                    | IntentionalBreak::ContentBasedWorldIdentity,
                    BreakPath::RecomputeFromSource { error, .. },
                ) => {
                    let refused = adopt_carried_target_digest(&[0; 32], &[1; 32]).unwrap_err();
                    assert!(error.ends_with("CarriedTargetDigestNotTranslated"));
                    assert_eq!(refused, MigrationError::CarriedTargetDigestNotTranslated);
                }
                (
                    IntentionalBreak::ReducedRetentionOperation,
                    BreakPath::Refused { error, next_attempt },
                ) => {
                    // The refusal is enforced in `antecedent_state`, which this crate does not
                    // depend on; the facade consumer test calls it.
                    assert!(error.ends_with("RetentionRefusal"));
                    assert!(next_attempt.contains("reload"));
                }
                (IntentionalBreak::RouteWithoutCurrentMethod, BreakPath::Refused { error, .. }) => {
                    let refused = migrate_route_record(&licensed, &[]).unwrap_err();
                    assert!(matches!(refused, RouteRecordError::RouteWithoutCurrentMethod { .. }));
                    assert!(error.ends_with("RouteWithoutCurrentMethod"));
                }
                (IntentionalBreak::LicenseNotCarried, BreakPath::Refused { error, .. }) => {
                    let current = [method("m", "r", "static", "bayesian", "execution_only")];
                    let refused = migrate_route_record(&licensed, &current).unwrap_err();
                    assert!(matches!(refused, RouteRecordError::LicenseNotCarried { .. }));
                    assert!(error.ends_with("LicenseNotCarried"));
                }
                (IntentionalBreak::RefusalReason, BreakPath::Unchanged { .. }) => {
                    assert_eq!(
                        keep_refusal_reason("policy_route_not_licensed"),
                        "policy_route_not_licensed"
                    );
                }
                (other, path) => panic!("{other:?} has the unexpected path {path:?}"),
            }
        }
    }
}
