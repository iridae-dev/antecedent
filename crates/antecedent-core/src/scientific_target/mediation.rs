//! Mediation targets and their cross-world premises.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::canonical_encode::{CanonicalEncode, count, encode_variables};
use crate::identity::CanonicalWriter;
use crate::ids::VariableId;
use crate::intervention::{Intervention, TemporalPolicy};
use std::collections::BTreeSet;
use std::sync::Arc;

use super::model::{
    TimeKind, canonical_intervention, ensure_policy_within_horizon, ensure_portable,
};
use super::{ActionFault, ScientificTargetError, TimeFrame, WorldBinding};

/// Which mediation functional is the target.
///
/// These names match [`crate::query::MediationContrast`]. [`Self::Direct`] is the
/// natural direct path-sum, not a controlled direct effect licensed by algebra.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum MediationComponent {
    /// Total effect. Cross-world premises may be empty; mediators are still part of the target.
    Total,
    /// Natural direct path-sum that avoids the declared mediators.
    Direct,
    /// Effect along the declared mediators.
    Mediated,
    /// Natural direct effect.
    NaturalDirect,
    /// Natural indirect effect.
    NaturalIndirect,
}

impl MediationComponent {
    /// Canonical tag. Exhaustive, so a new component must be numbered here.
    const fn canonical_tag(self) -> u64 {
        match self {
            Self::Total => 1,
            Self::Direct => 2,
            Self::Mediated => 3,
            Self::NaturalDirect => 4,
            Self::NaturalIndirect => 5,
        }
    }

    /// Whether algebra is not enough and cross-world premises are required.
    #[must_use]
    pub const fn requires_cross_world(self) -> bool {
        !matches!(self, Self::Total)
    }
}

/// Premise a mediation target must carry. A contrast does not have these.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[non_exhaustive]
pub enum CrossWorldPremise {
    /// Setting a variable to the value it would have taken does not further change descendants.
    Composition,
    /// The mediator's counterfactual under one treatment is independent of the other treatment world.
    CrossWorldIndependence,
    /// The functional is a nested path-specific effect.
    PathSpecificNested,
}

impl CrossWorldPremise {
    const fn tag(self) -> u64 {
        match self {
            Self::Composition => 1,
            Self::CrossWorldIndependence => 2,
            Self::PathSpecificNested => 3,
        }
    }
}

/// Treatment, mediators, and the two treatment interventions of a mediation target.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct MediationArms {
    treatment: VariableId,
    mediators: Arc<[VariableId]>,
    control: Intervention,
    active: Intervention,
    component: MediationComponent,
}

impl MediationArms {
    /// Arms. Mediators are sorted and de-duplicated.
    ///
    /// # Errors
    ///
    /// [`ScientificTargetError::EmptyMediators`], [`ScientificTargetError::MediatorOverlap`],
    /// [`ScientificTargetError::InterventionVariableMismatch`],
    /// [`ScientificTargetError::MediationArmsNotDistinct`],
    /// [`ScientificTargetError::InvalidAction`], or [`ScientificTargetError::CallbackAction`].
    pub fn new(
        treatment: VariableId,
        mediators: impl IntoIterator<Item = VariableId>,
        control: Intervention,
        active: Intervention,
        component: MediationComponent,
    ) -> Result<Self, ScientificTargetError> {
        let mut mediators: Vec<VariableId> = mediators.into_iter().collect();
        mediators.sort_unstable();
        mediators.dedup();
        if mediators.is_empty() {
            return Err(ScientificTargetError::EmptyMediators);
        }
        if mediators.contains(&treatment) {
            return Err(ScientificTargetError::MediatorOverlap);
        }
        let control = canonical_intervention(control);
        let active = canonical_intervention(active);
        expect_treatment(&control, treatment)?;
        expect_treatment(&active, treatment)?;
        if control == active {
            return Err(ScientificTargetError::MediationArmsNotDistinct);
        }
        Ok(Self { treatment, mediators: Arc::from(mediators), control, active, component })
    }

    /// Treatment variable.
    #[must_use]
    pub const fn treatment(&self) -> VariableId {
        self.treatment
    }

    /// Mediators, sorted.
    #[must_use]
    pub fn mediators(&self) -> &[VariableId] {
        &self.mediators
    }

    /// Reference treatment intervention.
    #[must_use]
    pub const fn control(&self) -> &Intervention {
        &self.control
    }

    /// Active treatment intervention.
    #[must_use]
    pub const fn active(&self) -> &Intervention {
        &self.active
    }

    /// Mediation component.
    #[must_use]
    pub const fn component(&self) -> MediationComponent {
        self.component
    }
}

fn expect_treatment(
    intervention: &Intervention,
    treatment: VariableId,
) -> Result<(), ScientificTargetError> {
    ensure_portable(intervention)?;
    match intervention.primary_variable() {
        Some(variable) if variable == treatment => Ok(()),
        Some(variable) => Err(ScientificTargetError::InterventionVariableMismatch {
            expected: treatment,
            got: variable,
        }),
        None => Err(ScientificTargetError::InvalidAction {
            reason: ActionFault::MediationTreatmentTarget,
        }),
    }
}

/// Mediation target. Not constructible from [`super::InterventionContrast`].
///
/// [`MediationComponent::Direct`], [`MediationComponent::Mediated`],
/// [`MediationComponent::NaturalDirect`], and [`MediationComponent::NaturalIndirect`]
/// require [`CrossWorldPremise::CrossWorldIndependence`] and
/// [`CrossWorldPremise::PathSpecificNested`]. The two natural components are defined
/// through the nested counterfactual `Y(a, M(a'))`, so they also require
/// [`CrossWorldPremise::Composition`]. [`MediationComponent::Total`] may
/// record an empty premise set; the mediator set still keeps it distinct from a
/// treatment contrast.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct MediationTarget {
    binding: WorldBinding,
    time: TimeFrame,
    schedule: Option<TemporalPolicy>,
    arms: MediationArms,
    premises: BTreeSet<CrossWorldPremise>,
}

impl MediationTarget {
    /// Mediation under `time`. `schedule` is required exactly when `time` is temporal.
    ///
    /// # Errors
    ///
    /// [`ScientificTargetError::MediationNeedsCrossWorldPremises`] when a cross-world
    /// component lacks its premises, [`ScientificTargetError::MismatchedActionTime`]
    /// when the schedule and the frame disagree, [`ScientificTargetError::TreatmentEqualsOutcome`]
    /// or [`ScientificTargetError::MediatorOverlap`] when the arms collide with the outcome,
    /// [`ScientificTargetError::ConditioningIncludesTreatment`] or
    /// [`ScientificTargetError::ConditioningIncludesMediator`] when the binding conditions
    /// on the treatment or a mediator,
    /// [`ScientificTargetError::InvalidAction`] when a temporal policy is invalid, and
    /// [`ScientificTargetError::PolicyOutsideHorizon`] when it reaches outside the horizon.
    pub fn new(
        binding: WorldBinding,
        time: TimeFrame,
        schedule: Option<TemporalPolicy>,
        arms: MediationArms,
        premises: impl IntoIterator<Item = CrossWorldPremise>,
    ) -> Result<Self, ScientificTargetError> {
        if let Some(policy) = &schedule {
            policy.validate().map_err(|err| ScientificTargetError::InvalidAction {
                reason: ActionFault::Intervention(err),
            })?;
            ensure_policy_within_horizon(policy, &time)?;
        }
        let schedule_matches = matches!(
            (&time.kind, schedule.is_some()),
            (TimeKind::Static, false) | (TimeKind::Temporal { .. }, true)
        );
        if !schedule_matches {
            return Err(ScientificTargetError::MismatchedActionTime);
        }
        let outcome = binding.outcome().outcome();
        if arms.treatment == outcome {
            return Err(ScientificTargetError::TreatmentEqualsOutcome { id: outcome });
        }
        if arms.mediators.contains(&outcome) {
            return Err(ScientificTargetError::MediatorOverlap);
        }
        for &id in binding.conditioning().variables() {
            if id == arms.treatment {
                return Err(ScientificTargetError::ConditioningIncludesTreatment { id });
            }
            if arms.mediators.contains(&id) {
                return Err(ScientificTargetError::ConditioningIncludesMediator { id });
            }
        }
        let premises: BTreeSet<CrossWorldPremise> = premises.into_iter().collect();
        let natural = matches!(
            arms.component,
            MediationComponent::NaturalDirect | MediationComponent::NaturalIndirect
        );
        if arms.component.requires_cross_world()
            && !(premises.contains(&CrossWorldPremise::CrossWorldIndependence)
                && premises.contains(&CrossWorldPremise::PathSpecificNested)
                && (!natural || premises.contains(&CrossWorldPremise::Composition)))
        {
            return Err(ScientificTargetError::MediationNeedsCrossWorldPremises);
        }
        Ok(Self { binding, time, schedule, arms, premises })
    }

    /// Population, unit, quantity, and conditioning.
    #[must_use]
    pub const fn binding(&self) -> &WorldBinding {
        &self.binding
    }

    /// Decision time.
    #[must_use]
    pub const fn time(&self) -> &TimeFrame {
        &self.time
    }

    /// Treatment schedule when the target is temporal.
    #[must_use]
    pub const fn schedule(&self) -> Option<&TemporalPolicy> {
        self.schedule.as_ref()
    }

    /// Treatment arms.
    #[must_use]
    pub const fn arms(&self) -> &MediationArms {
        &self.arms
    }

    /// Cross-world premises. Empty only for a total effect.
    #[must_use]
    pub const fn premises(&self) -> &BTreeSet<CrossWorldPremise> {
        &self.premises
    }
}

impl CanonicalEncode for MediationTarget {
    fn encode(&self, writer: &mut CanonicalWriter) {
        self.binding.encode(writer);
        self.time.encode(writer);
        match &self.schedule {
            None => {
                writer.word(0);
            }
            Some(policy) => {
                writer.word(1);
                policy.encode(writer);
            }
        }
        self.arms.treatment.encode(writer);
        encode_variables(writer, &self.arms.mediators);
        self.arms.control.encode(writer);
        self.arms.active.encode(writer);
        writer.word(self.arms.component.canonical_tag());
        let mut tags: Vec<u64> = self.premises.iter().map(|premise| premise.tag()).collect();
        tags.sort_unstable();
        writer.word(count(tags.len()));
        for tag in tags {
            writer.word(tag);
        }
    }
}
