//! `compose_claims` with `ClaimOperation::Retarget` relays one parent's verified content and
//! may only weaken positive domain statuses to unknown.
//!
//! Positive cases weaken positive statuses (identified, supported, evaluated) to unknown.
//! Negative cases change an axis to any other status, relay two parents, or alter the value,
//! units, identities, kind, execution, evidence or calibration the relay must carry unchanged.
//!
//! NOTE: the refusal reason for a raised domain depends on the target status. A target that is
//! itself positive and differs from the parent is refused earlier by the general upgrade check
//! as `derived_upgrades_domain`; every other disallowed change reaches the retarget-specific
//! check and is refused as `retarget_invalid_domain_change`. Both are refusals of the same
//! forbidden change; the tests assert the reason each path produces.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::{
    ClaimDomainAxis, ClaimDomains, ClaimEnvelope, ClaimKind, ClaimOperation, DerivedClaimOutcome,
    DomainStatus, ReasoningView, SlotAvailability, compose_claims,
};
use antecedent_core::{ContractIdentities, SemanticDigest};
use proptest::prelude::*;

fn config() -> ProptestConfig {
    ProptestConfig { cases: 256, failure_persistence: None, ..ProptestConfig::default() }
}

const STATUSES: [DomainStatus; 8] = [
    DomainStatus::Identified,
    DomainStatus::Supported,
    DomainStatus::Evaluated,
    DomainStatus::OutsideScope,
    DomainStatus::Contradicted,
    DomainStatus::Unknown,
    DomainStatus::Limited,
    DomainStatus::NotClaimed,
];

const AXES: [ClaimDomainAxis; 3] =
    [ClaimDomainAxis::Identification, ClaimDomainAxis::Support, ClaimDomainAxis::Evaluated];

fn status() -> impl Strategy<Value = DomainStatus> {
    prop::sample::select(STATUSES.to_vec())
}

fn axis() -> impl Strategy<Value = ClaimDomainAxis> {
    prop::sample::select(AXES.to_vec())
}

fn is_positive(status: DomainStatus) -> bool {
    matches!(status, DomainStatus::Identified | DomainStatus::Supported | DomainStatus::Evaluated)
}

fn digest(byte: u8) -> SemanticDigest {
    SemanticDigest::from_bytes([byte; 32])
}

fn domains(values: [DomainStatus; 3]) -> ClaimDomains {
    ClaimDomains::new(values[0], values[1], values[2])
}

fn set_domain(claim: &mut ClaimEnvelope, axis: ClaimDomainAxis, to: DomainStatus) {
    match axis {
        ClaimDomainAxis::Identification => claim.domains.identification = to,
        ClaimDomainAxis::Support => claim.domains.support = to,
        ClaimDomainAxis::Evaluated => claim.domains.evaluated = to,
        _ => unreachable!("only three axes exist"),
    }
}

/// A parent claim with a shared identification product, so a retarget is compatible.
fn parent(id: u8, value: f64, values: [DomainStatus; 3]) -> ClaimEnvelope {
    ClaimEnvelope::new(
        digest(id),
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
        Some(value),
        Some(Arc::from("units")),
        ReasoningView::new(
            SlotAvailability::unavailable("fixture"),
            SlotAvailability::unavailable("fixture"),
            SlotAvailability::unavailable("fixture"),
            SlotAvailability::unavailable("fixture"),
        ),
        domains(values),
        None,
        [Arc::<str>::from("prior:p0"), Arc::<str>::from("joint_draws:draw-set-1")],
    )
}

fn statuses() -> impl Strategy<Value = [DomainStatus; 3]> {
    [status(), status(), status()]
}

fn refusal(outcome: &DerivedClaimOutcome) -> Option<&str> {
    match outcome {
        DerivedClaimOutcome::Refused { restriction, .. } => Some(&**restriction),
        _ => None,
    }
}

proptest! {
    #![proptest_config(config())]

    /// A relay equal to its parent except that some positive statuses became unknown is
    /// derived, carries the weakened domains and the parent's id, and keeps the parent value.
    #[test]
    fn relay_weakening_is_accepted(
        values in statuses(),
        weaken in prop::array::uniform3(any::<bool>()),
        value in -1.0e6f64..1.0e6,
        id in 20u8..200,
    ) {
        let parent = parent(id, value, values);
        let mut relayed = values;
        for index in 0..3 {
            if weaken[index] && is_positive(values[index]) {
                relayed[index] = DomainStatus::Unknown;
            }
        }
        let mut derived = parent.clone();
        derived.domains = domains(relayed);
        let outcome = compose_claims(&[&parent], ClaimOperation::Retarget, Some(derived), None);
        match outcome {
            DerivedClaimOutcome::Derived(claim) => {
                prop_assert_eq!(claim.claim.domains, domains(relayed));
                prop_assert_eq!(claim.claim.value.map(f64::to_bits), Some(value.to_bits()));
                prop_assert_eq!(&*claim.parents, &[parent.claim_id][..]);
            }
            other => prop_assert!(false, "weakening was not accepted: {other:?}"),
        }
    }

    /// A relay changing one axis from `from` to `to` is accepted exactly when the status is
    /// unchanged or a positive status is weakened to unknown; every other change is refused.
    #[test]
    fn relay_raising_domain_is_refused(
        axis in axis(),
        from in status(),
        to in status(),
        others in statuses(),
    ) {
        let mut parent = parent(30, 1.5, others);
        set_domain(&mut parent, axis, from);
        let mut derived = parent.clone();
        set_domain(&mut derived, axis, to);
        let allowed = from == to || (is_positive(from) && to == DomainStatus::Unknown);
        let outcome = compose_claims(&[&parent], ClaimOperation::Retarget, Some(derived), None);
        if allowed {
            prop_assert!(
                matches!(outcome, DerivedClaimOutcome::Derived(_)),
                "allowed change {from:?} -> {to:?} refused: {outcome:?}"
            );
        } else {
            let expected = if is_positive(to) {
                "derived_upgrades_domain"
            } else {
                "retarget_invalid_domain_change"
            };
            prop_assert_eq!(
                refusal(&outcome),
                Some(expected),
                "{:?} -> {:?} on {:?}",
                from,
                to,
                axis
            );
        }
    }

    /// A relay with two parents is refused, and a relay whose value, units, identities, kind,
    /// execution, evidence or calibration differ from its parent's is refused as changed.
    #[test]
    fn relay_without_its_premises_is_refused(
        values in statuses(),
        value in -1.0e6f64..1.0e6,
        mutation in 0usize..7,
    ) {
        let parent = parent(40, value, values);

        let mut second = parent.clone();
        second.claim_id = digest(41);
        second.value = Some(value + 1.0);
        let two = compose_claims(
            &[&parent, &second],
            ClaimOperation::Retarget,
            Some(parent.clone()),
            None,
        );
        prop_assert_eq!(refusal(&two), Some("retarget_requires_one_parent"));

        let mut derived = parent.clone();
        match mutation {
            0 => derived.value = Some(value + 1.0),
            1 => derived.outcome_units = Some(Arc::from("other_units")),
            2 => derived.identities.data_snapshot = digest(99),
            3 => derived.kind = ClaimKind::Bounds,
            4 => derived.execution = Some(digest(98)),
            5 => derived.evidence = Arc::from([Arc::<str>::from("prior:other")]),
            _ => derived.calibration.status = Arc::from("calibrated"),
        }
        let outcome = compose_claims(&[&parent], ClaimOperation::Retarget, Some(derived), None);
        prop_assert_eq!(refusal(&outcome), Some("retarget_requires_unchanged_parent"));
    }
}
