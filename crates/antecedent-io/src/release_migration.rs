//! Migration from the previous release onto the current contracts.
//!
//! Route axes stay unmigrated until a later change says otherwise. A carried
//! refusal reason is not rewritten. A score-row digest is recomputed from wide
//! row ids, and a digest that does not match is refused. A carried scientific
//! target digest is refused unless it is already the current digest. Reduced
//! retention already refuses later operations through
//! `antecedent_state::retention::RetentionMode`. This module does not declare
//! a second retention type.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::PayloadDigestWire;

/// Kind string of the score-table row-index digest.
pub const SCORE_ROW_IDENTITY_KIND: &str = "score_reuse.row_index";

/// One migration child: a time frame crossed with an inference realization.
///
/// A geometric cell is not a child.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
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

/// Whether licensed and graphless routes on an axis are on the current contracts.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum AxisStanding {
    /// Those routes are on the current contracts.
    ///
    /// No axis returns this.
    #[allow(
        dead_code,
        reason = "no route axis is migrated; the variant keeps the classification closed"
    )]
    Migrated,
    /// Those routes are not yet on the current contracts.
    StillUnmigrated,
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
    /// Licensed and graphless routes on this axis are not on the current contracts.
    #[error(
        "{axis} is still unmigrated: licensed and graphless routes are not yet on the current \
         contracts"
    )]
    AxisStillUnmigrated {
        /// Axis that was asked.
        axis: RouteAxis,
    },
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

/// Standing of one route axis.
///
/// Every axis is still unmigrated. This does not relabel a route as migrated.
#[must_use]
pub const fn axis_standing(axis: RouteAxis) -> AxisStanding {
    match axis {
        RouteAxis::StaticFrequentist
        | RouteAxis::StaticBayesian
        | RouteAxis::TemporalFrequentist
        | RouteAxis::TemporalBayesian => AxisStanding::StillUnmigrated,
    }
}

/// Standing of one manifest route.
///
/// Licensed and graphless routes are not on the current contracts. Closed,
/// execution-only, and fitting-gap routes are not relabelled as migrated.
#[must_use]
pub const fn route_contract_standing(status: ManifestRouteStatus) -> AxisStanding {
    match status {
        ManifestRouteStatus::Licensed
        | ManifestRouteStatus::Graphless
        | ManifestRouteStatus::Closed
        | ManifestRouteStatus::ExecutionOnly
        | ManifestRouteStatus::FittingGap => AxisStanding::StillUnmigrated,
    }
}

/// Migrate one axis onto the current contracts.
///
/// # Errors
///
/// The axis is still unmigrated.
#[must_use = "an unmigrated axis is refused and was not relabelled"]
pub const fn migrate_axis(axis: RouteAxis) -> Result<(), MigrationError> {
    match axis_standing(axis) {
        AxisStanding::Migrated => Ok(()),
        AxisStanding::StillUnmigrated => Err(MigrationError::AxisStillUnmigrated { axis }),
    }
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

#[cfg(test)]
mod tests {
    use antecedent_core::{
        AssumptionSet, Conditioning, GraphFamily, InferenceSelection, Intervention,
        InterventionContrast, InterventionWorld, ModelAssumptions, ObservationUnit,
        OutcomeQuantity, QuantityUnits, ScientificTarget, ScientificWorldId, TargetForm,
        TargetPopulation, TimeFrame, Value, VariableId, WorldAction, WorldBinding, WorldCollection,
    };

    use super::{
        AxisStanding, ManifestRouteStatus, MigrationError, RouteAxis, TargetShape,
        adopt_carried_target_digest, adopt_retained_score_row_digest, axis_standing,
        caller_world_id_and_reference_flag_enter_identity, current_score_row_digest,
        keep_refusal_reason, migrate_axis, narrow_score_row_digest, route_contract_standing,
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
    fn every_route_axis_is_still_unmigrated() {
        assert_eq!(RouteAxis::ALL.len(), 4);
        for axis in RouteAxis::ALL {
            assert_eq!(axis_standing(axis), AxisStanding::StillUnmigrated);
            let error = migrate_axis(axis).unwrap_err();
            assert_eq!(error, MigrationError::AxisStillUnmigrated { axis });
            assert!(error.to_string().contains(axis.as_str()));
            assert!(error.to_string().contains("not yet on the current contracts"));
        }
    }

    #[test]
    fn licensed_and_graphless_routes_are_not_migrated() {
        for status in [ManifestRouteStatus::Licensed, ManifestRouteStatus::Graphless] {
            assert_eq!(route_contract_standing(status), AxisStanding::StillUnmigrated);
        }
    }

    #[test]
    fn closed_execution_and_fitting_routes_are_not_relabelled_migrated() {
        for status in ManifestRouteStatus::ALL {
            assert_eq!(route_contract_standing(status), AxisStanding::StillUnmigrated);
        }
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
        let other = target(TargetForm::World(world(2, true, 0.0))).identity();
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
}
