// SPDX-License-Identifier: MIT OR Apache-2.0

//! Input contract for experiment regimes handed to a later clamp-experiment recipe.
//!
//! Each [`ExperimentRegime`] declares an intervention target set and the value of
//! each target, the sampling units, the population, an independence or dependence
//! premise, and a source identity. [`validate_regime_contract`] accepts that
//! declaration or refuses it before any recovery. It does not call
//! [`decide_experimental_recovery`](crate::experimental_recovery::decide_experimental_recovery),
//! read a sample, or fit a mechanism.
//!
//! A [`UnitMask`] that drops a targeted unit is refused. A shared source identity
//! contradicts a [`DependencePremise::Independence`] premise on either regime of
//! a pair; that pair is refused. A [`DependencePremise::Dependence`] premise
//! agrees with a shared source and is not, by itself, a contradiction. A
//! dependence premise on a regime that shares its source with no other regime
//! claims a shared disturbance with nothing, and is refused. A target
//! outside the regime's sampling units is an incompatible regime and is refused.
//! Every refusal names [`NextRegimeAttempt::RepairMaskOrPremise`]: repair the
//! mask or the premise, and do not fit the mechanism on the illegal regime.
//!
//! The declarations are inputs a later
//! [`DeclaredClampExperiment`](crate::experimental_recovery::DeclaredClampExperiment)
//! and [`RegimeAssumption`](crate::experimental_recovery::RegimeAssumption) can be
//! built from. [`DependencePremise::Independence`] is the input-side
//! counterpart of
//! [`RegimeAssumption::Independent`](crate::experimental_recovery::RegimeAssumption::Independent);
//! [`DependencePremise::Dependence`] is the counterpart of
//! [`RegimeAssumption::Dependent`](crate::experimental_recovery::RegimeAssumption::Dependent).
//! This check does not apply that recipe's exogeneity, noise-law, coverage, rank,
//! or weak-stability conditions, and it does not produce a
//! [`CoefficientRecovery`](crate::experimental_recovery::CoefficientRecovery).
//! A dependence premise this contract accepts may still be refused later by the
//! recipe; this function does not make that call.
//!
//! The contract extends Hyttinen, Eberhardt, and Hoyer (JMLR 13, `hyttinen12a`)
//! only as an input gate, not as their estimator. It does not inherit
//! identifiability of a linear cyclic model with latent variables, recovery of
//! the direct-effect matrix from experimental covariances, their learning
//! procedure or its rank tests, or a weak-stability or consistency theorem.

use std::collections::BTreeSet;
use std::fmt;

use antecedent_core::{Value, VariableId};

/// Stable name of the repair a refused regime points at.
///
/// This is not [`CLAMP_EXPERIMENT_RECIPE`](crate::experimental_recovery::CLAMP_EXPERIMENT_RECIPE)
/// and it is not a mechanism fit.
pub const REPAIR_MASK_OR_PREMISE: &str = "experiment_regime.repair_mask_or_premise";

/// Caller-assigned identity of the sample a regime is drawn from.
///
/// Equal values are the same source. The caller supplies the identity. Assembly
/// refuses a sample whose recorded source is not this identity. Zero is a real
/// identity, not an unset marker.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SourceIdentity(u64);

impl SourceIdentity {
    /// Identity `raw`.
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Underlying integer.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

impl fmt::Display for SourceIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "source:{}", self.0)
    }
}

/// Caller-assigned identity of the population a regime's units are drawn from.
///
/// This module does not transport a law across populations.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct PopulationId(u32);

impl PopulationId {
    /// Identity `raw`. Zero is a real population identity.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Underlying integer.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// Whether a regime's disturbance is claimed to be a fresh draw.
///
/// This is the input premise. It is not a test of the noise law and it is not
/// [`RegimeAssumption`](crate::experimental_recovery::RegimeAssumption) applied
/// inside the recipe.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DependencePremise {
    /// The regime is claimed independent of every other regime in the contract.
    ///
    /// A shared [`SourceIdentity`] contradicts this premise.
    Independence,
    /// The regime is claimed to share its disturbance with another regime.
    ///
    /// A shared [`SourceIdentity`] agrees with this premise.
    Dependence,
}

/// One clamp: a sampling unit and the value that replaces its equation.
///
/// The value is the declared intervention level. It is not an estimated effect.
/// For a randomized clamp the level is a label for the arm, or the mean of the
/// randomized draws. It is one constant per target, and it does not say what the
/// clamped column looks like in the sample: the moment assembly needs that column
/// to vary across rows, and reads it from the sample, never from this value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegimeIntervention {
    /// Sampling unit the clamp replaces.
    pub unit: VariableId,
    /// Declared level of that clamp.
    pub value: Value,
}

/// One experiment regime offered to a later clamp-experiment recipe.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExperimentRegime {
    /// Intervention target set. Each target carries its clamp value.
    pub targets: Vec<RegimeIntervention>,
    /// Sampling units. Every target must be one of these units.
    pub units: Vec<VariableId>,
    /// Population the sampling units are drawn from.
    pub population: PopulationId,
    /// Independence or dependence premise for this regime.
    pub premise: DependencePremise,
    /// Source the regime's sample is drawn from.
    pub source: SourceIdentity,
}

/// Units removed before a later recipe sees the regimes.
///
/// Dropping a unit that a regime targets is illegal. Dropping a declared unit
/// that no regime targets is allowed.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct UnitMask {
    /// Sampling units the mask removes.
    pub dropped: Vec<VariableId>,
}

/// Regimes plus the mask judged against them.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegimeContract {
    /// Declared regimes, in caller order.
    pub regimes: Vec<ExperimentRegime>,
    /// Mask applied to those regimes.
    pub mask: UnitMask,
}

/// Why the contract was refused before recovery.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RegimeRefusalReason {
    /// The contract listed no regime.
    NoRegime,
    /// A regime listed no sampling unit.
    EmptySamplingUnits {
        /// Index into [`RegimeContract::regimes`].
        regime: usize,
    },
    /// The same sampling unit was listed twice in one regime.
    DuplicateSamplingUnit {
        /// Index into [`RegimeContract::regimes`].
        regime: usize,
        /// Repeated unit.
        unit: VariableId,
    },
    /// The same unit was targeted twice in one regime.
    DuplicateInterventionTarget {
        /// Index into [`RegimeContract::regimes`].
        regime: usize,
        /// Repeated target.
        unit: VariableId,
    },
    /// A target's value is not a concrete intervention level.
    ///
    /// Non-finite floats and labels, including the symbolic intervention marker,
    /// are not clamp values.
    NonConcreteInterventionValue {
        /// Index into [`RegimeContract::regimes`].
        regime: usize,
        /// Target whose value was refused.
        unit: VariableId,
    },
    /// A target is not a member of that regime's sampling units.
    TargetOutsideUnits {
        /// Index into [`RegimeContract::regimes`].
        regime: usize,
        /// Target that lies outside the unit set.
        unit: VariableId,
    },
    /// The mask removes a unit some regime targets.
    MaskDropsTargetedUnit {
        /// Index of the first regime that targets `unit`.
        regime: usize,
        /// Targeted unit the mask removes.
        unit: VariableId,
    },
    /// The mask names a unit no regime declares.
    MaskDropsUnknownUnit {
        /// Unit absent from every regime's sampling units.
        unit: VariableId,
    },
    /// A regime claims dependence but no other regime shares its source.
    DependenceWithoutSharedSource {
        /// Index into [`RegimeContract::regimes`].
        regime: usize,
        /// Source identity the regime declares.
        source: SourceIdentity,
    },
    /// Two regimes share a source while at least one claims independence.
    IndependenceContradictedBySharedSource {
        /// Index of the earlier regime.
        left: usize,
        /// Index of the later regime.
        right: usize,
        /// Source identity both regimes declare.
        source: SourceIdentity,
    },
}

impl fmt::Display for RegimeRefusalReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoRegime => write!(f, "no regime was declared"),
            Self::EmptySamplingUnits { regime } => {
                write!(f, "regime {regime} declares no sampling units")
            }
            Self::DuplicateSamplingUnit { regime, unit } => {
                write!(f, "regime {regime} repeats sampling unit {unit}")
            }
            Self::DuplicateInterventionTarget { regime, unit } => {
                write!(f, "regime {regime} repeats intervention target {unit}")
            }
            Self::NonConcreteInterventionValue { regime, unit } => {
                write!(f, "regime {regime} assigns a non-concrete value to {unit}")
            }
            Self::TargetOutsideUnits { regime, unit } => {
                write!(f, "regime {regime} targets {unit}, which is outside its sampling units")
            }
            Self::MaskDropsTargetedUnit { regime, unit } => {
                write!(f, "mask drops targeted unit {unit} in regime {regime}")
            }
            Self::MaskDropsUnknownUnit { unit } => {
                write!(f, "mask drops {unit}, which is not a declared sampling unit")
            }
            Self::DependenceWithoutSharedSource { regime, source } => {
                write!(f, "regime {regime} claims dependence but no other regime shares {source}")
            }
            Self::IndependenceContradictedBySharedSource { left, right, source } => {
                write!(f, "regimes {left} and {right} share {source}, contradicting independence",)
            }
        }
    }
}

/// The attempt a refusal points to.
///
/// The attempt is to repair the declaration. It is not a fit of the mechanism.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum NextRegimeAttempt {
    /// Repair the mask or the premise. Do not fit the mechanism on the illegal regime.
    RepairMaskOrPremise,
}

impl NextRegimeAttempt {
    /// Stable name of the repair.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::RepairMaskOrPremise => REPAIR_MASK_OR_PREMISE,
        }
    }
}

impl fmt::Display for NextRegimeAttempt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RepairMaskOrPremise => f.write_str("repair the mask or the premise"),
        }
    }
}

/// A contract refused before recovery.
///
/// `next_attempt` is always [`NextRegimeAttempt::RepairMaskOrPremise`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RegimeRefusal {
    /// Why the contract was refused.
    pub reason: RegimeRefusalReason,
    /// What to attempt next. Not a mechanism fit.
    pub next_attempt: NextRegimeAttempt,
}

impl fmt::Display for RegimeRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}; next attempt: {} ({}); do not fit the mechanism on the illegal regime",
            self.reason,
            self.next_attempt,
            self.next_attempt.name(),
        )
    }
}

/// Legal regimes and mask.
///
/// This is an input license for a later recipe. It is not a coefficient
/// recovery, and it carries no mechanism matrix.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AcceptedRegimeContract {
    regimes: Vec<ExperimentRegime>,
    mask: UnitMask,
}

impl AcceptedRegimeContract {
    /// Regimes the contract accepted, in the submitted order.
    #[must_use = "the accepted regimes are the only legal input"]
    pub fn regimes(&self) -> &[ExperimentRegime] {
        &self.regimes
    }

    /// Mask the contract accepted. It does not drop a targeted unit.
    #[must_use = "the accepted mask is part of the legal input"]
    pub const fn mask(&self) -> &UnitMask {
        &self.mask
    }
}

/// Outcome of judging a regime contract before recovery.
#[derive(Clone, Debug, Eq, PartialEq)]
#[must_use = "a refused regime must not be passed to mechanism recovery"]
pub enum RegimeContractDecision {
    /// The regimes and the mask are legal inputs.
    Accepted(AcceptedRegimeContract),
    /// The contract is illegal. Do not fit a mechanism on it.
    Refused(RegimeRefusal),
}

impl RegimeContractDecision {
    /// Accepted contract, when the regimes and the mask are legal inputs.
    #[must_use = "absence means the regime was refused before recovery"]
    pub const fn accepted(&self) -> Option<&AcceptedRegimeContract> {
        match self {
            Self::Accepted(contract) => Some(contract),
            Self::Refused(_) => None,
        }
    }
}

/// Judge `contract` before any clamp-experiment recovery.
///
/// The function inspects the declared targets, sampling units, populations,
/// premises, source identities, and mask. It does not estimate effects, invert
/// a constraint, or complete a mechanism matrix. A refusal's next attempt is
/// [`NextRegimeAttempt::RepairMaskOrPremise`].
#[must_use = "a refusal must be repaired before any mechanism is fit"]
pub fn validate_regime_contract(contract: &RegimeContract) -> RegimeContractDecision {
    if let Some(reason) = contract_fault(contract) {
        return refuse(reason);
    }
    RegimeContractDecision::Accepted(AcceptedRegimeContract {
        regimes: contract.regimes.clone(),
        mask: contract.mask.clone(),
    })
}

fn refuse(reason: RegimeRefusalReason) -> RegimeContractDecision {
    RegimeContractDecision::Refused(RegimeRefusal {
        reason,
        next_attempt: NextRegimeAttempt::RepairMaskOrPremise,
    })
}

fn contract_fault(contract: &RegimeContract) -> Option<RegimeRefusalReason> {
    if contract.regimes.is_empty() {
        return Some(RegimeRefusalReason::NoRegime);
    }
    for (index, regime) in contract.regimes.iter().enumerate() {
        if let Some(reason) = regime_fault(index, regime) {
            return Some(reason);
        }
    }
    if let Some(reason) = mask_fault(&contract.regimes, &contract.mask) {
        return Some(reason);
    }
    shared_source_fault(&contract.regimes)
}

fn regime_fault(index: usize, regime: &ExperimentRegime) -> Option<RegimeRefusalReason> {
    if regime.units.is_empty() {
        return Some(RegimeRefusalReason::EmptySamplingUnits { regime: index });
    }
    let mut units = BTreeSet::new();
    for &unit in &regime.units {
        if !units.insert(unit) {
            return Some(RegimeRefusalReason::DuplicateSamplingUnit { regime: index, unit });
        }
    }
    let mut targets = BTreeSet::new();
    for target in &regime.targets {
        if !targets.insert(target.unit) {
            return Some(RegimeRefusalReason::DuplicateInterventionTarget {
                regime: index,
                unit: target.unit,
            });
        }
        if !units.contains(&target.unit) {
            return Some(RegimeRefusalReason::TargetOutsideUnits {
                regime: index,
                unit: target.unit,
            });
        }
        if target.value.validate_concrete_intervention_level().is_err() {
            return Some(RegimeRefusalReason::NonConcreteInterventionValue {
                regime: index,
                unit: target.unit,
            });
        }
    }
    None
}

fn mask_fault(regimes: &[ExperimentRegime], mask: &UnitMask) -> Option<RegimeRefusalReason> {
    for &unit in &mask.dropped {
        let mut known = false;
        for (regime, experiment) in regimes.iter().enumerate() {
            if experiment.units.contains(&unit) {
                known = true;
            }
            if experiment.targets.iter().any(|target| target.unit == unit) {
                return Some(RegimeRefusalReason::MaskDropsTargetedUnit { regime, unit });
            }
        }
        if !known {
            return Some(RegimeRefusalReason::MaskDropsUnknownUnit { unit });
        }
    }
    None
}

/// A shared source agrees with dependence. It contradicts an independence premise.
/// A dependence premise needs another regime on the same source.
fn shared_source_fault(regimes: &[ExperimentRegime]) -> Option<RegimeRefusalReason> {
    for left in 0..regimes.len() {
        for right in (left + 1)..regimes.len() {
            let same_source = regimes[left].source == regimes[right].source;
            let independence = regimes[left].premise == DependencePremise::Independence
                || regimes[right].premise == DependencePremise::Independence;
            if same_source && independence {
                return Some(RegimeRefusalReason::IndependenceContradictedBySharedSource {
                    left,
                    right,
                    source: regimes[left].source,
                });
            }
        }
    }
    for (index, regime) in regimes.iter().enumerate() {
        let shared = regimes
            .iter()
            .enumerate()
            .any(|(other, candidate)| other != index && candidate.source == regime.source);
        if regime.premise == DependencePremise::Dependence && !shared {
            return Some(RegimeRefusalReason::DependenceWithoutSharedSource {
                regime: index,
                source: regime.source,
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{
        DependencePremise, ExperimentRegime, NextRegimeAttempt, PopulationId,
        REPAIR_MASK_OR_PREMISE, RegimeContract, RegimeContractDecision, RegimeIntervention,
        RegimeRefusalReason, SourceIdentity, UnitMask, validate_regime_contract,
    };
    use antecedent_core::{Value, VariableId};

    fn unit(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    fn intervention(unit_raw: u32, value: i64) -> RegimeIntervention {
        RegimeIntervention { unit: unit(unit_raw), value: Value::Int64(value) }
    }

    fn regime(
        units: &[u32],
        targets: &[RegimeIntervention],
        premise: DependencePremise,
        source: u64,
    ) -> ExperimentRegime {
        ExperimentRegime {
            targets: targets.to_vec(),
            units: units.iter().copied().map(unit).collect(),
            population: PopulationId::from_raw(1),
            premise,
            source: SourceIdentity::from_raw(source),
        }
    }

    fn contract(regimes: Vec<ExperimentRegime>, dropped: &[u32]) -> RegimeContract {
        RegimeContract {
            regimes,
            mask: UnitMask { dropped: dropped.iter().copied().map(unit).collect() },
        }
    }

    fn assert_refused(decision: &RegimeContractDecision, reason: RegimeRefusalReason) {
        assert!(decision.accepted().is_none());
        let RegimeContractDecision::Refused(refusal) = decision else {
            panic!("expected a refusal before recovery");
        };
        assert_eq!(refusal.reason, reason);
        assert_eq!(refusal.next_attempt, NextRegimeAttempt::RepairMaskOrPremise);
        assert_eq!(refusal.next_attempt.name(), REPAIR_MASK_OR_PREMISE);
        let message = refusal.to_string();
        assert!(message.contains("repair the mask or the premise"), "{message}");
        assert!(message.contains("do not fit the mechanism"), "{message}");
    }

    #[test]
    fn complete_regime_is_accepted() {
        let submitted = regime(&[0, 1], &[intervention(0, 1)], DependencePremise::Independence, 9);
        let decision = validate_regime_contract(&contract(vec![submitted.clone()], &[]));
        let RegimeContractDecision::Accepted(accepted) = &decision else {
            panic!("expected a complete regime to be accepted");
        };
        assert_eq!(accepted.regimes(), &[submitted]);
        assert!(accepted.mask().dropped.is_empty());
        assert_eq!(accepted.regimes()[0].population, PopulationId::from_raw(1));
        assert_eq!(accepted.regimes()[0].premise, DependencePremise::Independence);
        assert_eq!(accepted.regimes()[0].source, SourceIdentity::from_raw(9));
    }

    #[test]
    fn mask_that_removes_a_targeted_unit_fails() {
        let submitted = regime(&[0, 1], &[intervention(0, 1)], DependencePremise::Independence, 3);
        let decision = validate_regime_contract(&contract(vec![submitted], &[0]));
        assert_refused(
            &decision,
            RegimeRefusalReason::MaskDropsTargetedUnit { regime: 0, unit: unit(0) },
        );
    }

    #[test]
    fn shared_source_plus_an_independence_premise_fails() {
        let left = regime(&[0, 1], &[intervention(0, 1)], DependencePremise::Independence, 4);
        let right = regime(&[0, 1], &[intervention(1, 2)], DependencePremise::Independence, 4);
        let decision = validate_regime_contract(&contract(vec![left, right], &[]));
        assert_refused(
            &decision,
            RegimeRefusalReason::IndependenceContradictedBySharedSource {
                left: 0,
                right: 1,
                source: SourceIdentity::from_raw(4),
            },
        );
    }

    #[test]
    fn target_outside_the_units_fails() {
        let submitted = regime(&[0, 1], &[intervention(2, 1)], DependencePremise::Dependence, 5);
        let decision = validate_regime_contract(&contract(vec![submitted], &[]));
        assert_refused(
            &decision,
            RegimeRefusalReason::TargetOutsideUnits { regime: 0, unit: unit(2) },
        );
    }

    #[test]
    fn one_independence_premise_with_a_shared_source_fails() {
        let left = regime(&[0, 1], &[intervention(0, 1)], DependencePremise::Independence, 4);
        let right = regime(&[0, 1], &[intervention(1, 2)], DependencePremise::Dependence, 4);
        let decision = validate_regime_contract(&contract(vec![left, right], &[]));
        assert_refused(
            &decision,
            RegimeRefusalReason::IndependenceContradictedBySharedSource {
                left: 0,
                right: 1,
                source: SourceIdentity::from_raw(4),
            },
        );
    }

    #[test]
    fn shared_source_with_a_dependence_premise_is_accepted() {
        let left = regime(&[0, 1], &[intervention(0, 1)], DependencePremise::Dependence, 4);
        let right = regime(&[0, 1], &[intervention(1, 2)], DependencePremise::Dependence, 4);
        let decision = validate_regime_contract(&contract(vec![left.clone(), right.clone()], &[]));
        let RegimeContractDecision::Accepted(accepted) = &decision else {
            panic!("a shared source agrees with a dependence premise");
        };
        assert_eq!(accepted.regimes(), &[left, right]);
    }

    #[test]
    fn a_dependence_premise_without_a_shared_source_fails() {
        let alone = regime(&[0, 1], &[intervention(0, 1)], DependencePremise::Dependence, 6);
        let other = regime(&[0, 1], &[intervention(1, 2)], DependencePremise::Dependence, 7);
        let decision = validate_regime_contract(&contract(vec![alone, other], &[]));
        assert_refused(
            &decision,
            RegimeRefusalReason::DependenceWithoutSharedSource {
                regime: 0,
                source: SourceIdentity::from_raw(6),
            },
        );
        // A dependence regime beside an independent regime on a distinct source claims
        // dependence with nothing.
        let dependent = regime(&[0, 1], &[intervention(0, 1)], DependencePremise::Dependence, 6);
        let independent =
            regime(&[0, 1], &[intervention(1, 2)], DependencePremise::Independence, 8);
        let decision = validate_regime_contract(&contract(vec![dependent, independent], &[]));
        assert_refused(
            &decision,
            RegimeRefusalReason::DependenceWithoutSharedSource {
                regime: 0,
                source: SourceIdentity::from_raw(6),
            },
        );
    }

    #[test]
    fn mask_may_drop_an_untargeted_unit() {
        let submitted = regime(&[0, 1], &[intervention(0, 1)], DependencePremise::Independence, 3);
        let decision = validate_regime_contract(&contract(vec![submitted.clone()], &[1]));
        let RegimeContractDecision::Accepted(accepted) = &decision else {
            panic!("a mask may drop a sampling unit that is not targeted");
        };
        assert_eq!(accepted.regimes(), &[submitted]);
        assert_eq!(accepted.mask().dropped, vec![unit(1)]);
    }

    #[test]
    fn distinct_sources_may_claim_independence() {
        let left = regime(&[0, 1], &[intervention(0, 1)], DependencePremise::Independence, 1);
        let right = regime(&[0, 1], &[intervention(1, 2)], DependencePremise::Independence, 2);
        let decision = validate_regime_contract(&contract(vec![left.clone(), right.clone()], &[]));
        let RegimeContractDecision::Accepted(accepted) = &decision else {
            panic!("distinct sources do not contradict independence");
        };
        assert_eq!(accepted.regimes(), &[left, right]);
    }
}
