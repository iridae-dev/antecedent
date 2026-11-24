//! Standing laws of the versioned recalculation planner and its receipts, over generated
//! workflows.
//!
//! The removal-first rule never leaves a stage in a better class than the first-change rule
//! did, the two rules agree whenever no input was removed, and a sealed receipt records the
//! rule it was planned under and replays exactly under it.

use std::collections::BTreeMap;

use antecedent_core::recalc::{
    PlannerRule, RecalcCapabilities, RecalcPlan, RetargetSupport, Stage, StageIdentities,
    StageIdentity, StageStatus,
};
use antecedent_io_core::recalc_receipt_artifact::{
    CountsWire, RECALC_RECEIPT_ARTIFACT_VERSION, RECALC_RECEIPT_REMOVAL_FIRST_VERSION,
    RecalcReceiptArtifact, planner_rule_of_version,
};
use proptest::prelude::*;

fn config() -> ProptestConfig {
    ProptestConfig { cases: 256, failure_persistence: None, ..ProptestConfig::default() }
}

const RETARGETS: [RetargetSupport; 3] =
    [RetargetSupport::Supported, RetargetSupport::NotDeclared, RetargetSupport::Incompatible];

/// Per stage: its version before (0 = not declared), whether the request keeps it, and its
/// version after (0 = removed).
fn workflow() -> impl Strategy<Value = Vec<(u8, bool, u8)>> {
    prop::collection::vec((0u8..3, prop::bool::weighted(0.6), 0u8..3), Stage::all().len())
}

fn identities(states: &[(u8, bool, u8)], requested: bool) -> StageIdentities {
    let mut out = StageIdentities::new();
    for (stage, &(before, keep, after)) in Stage::all().into_iter().zip(states) {
        let version = if requested && !keep { after } else { before };
        if version != 0 {
            out.set(stage, StageIdentity::of(&stage.label(), &[&[version][..]]));
        }
    }
    out
}

fn any_removed(previous: &StageIdentities, requested: &StageIdentities) -> bool {
    Stage::all().into_iter().any(|stage| previous.contains(stage) && !requested.contains(stage))
}

/// The work each recomputed computation must count.
fn work(plan: &RecalcPlan) -> BTreeMap<Stage, CountsWire> {
    plan.entries()
        .iter()
        .filter(|entry| matches!(entry.status, StageStatus::Recomputed { .. }))
        .filter_map(|entry| {
            let counts = match entry.stage {
                Stage::Identification => CountsWire { identifications: 1, ..CountsWire::default() },
                Stage::ScoreArtifact => {
                    CountsWire { fold_fits: 1, score_computations: 1, ..CountsWire::default() }
                }
                Stage::Law => CountsWire { reweights: 1, ..CountsWire::default() },
                Stage::Decision => CountsWire { decisions: 1, ..CountsWire::default() },
                Stage::ProviderRequest(_) => {
                    CountsWire { external_invocations: 1, ..CountsWire::default() }
                }
                _ => return None,
            };
            Some((entry.stage, counts))
        })
        .collect()
}

proptest! {
    #![proptest_config(config())]

    /// The removal-first rule never raises a stage's class over the first-change rule, and the
    /// rules give the same table whenever no input was removed.
    #[test]
    fn removal_first_never_raises_over_first_change(states in workflow(), retarget in 0usize..3) {
        let caps = RecalcCapabilities::in_process(RETARGETS[retarget]);
        let previous = identities(&states, false);
        let requested = identities(&states, true);
        let current = RecalcPlan::plan(&previous, &requested, &caps);
        let legacy = RecalcPlan::plan_with_rule(&previous, &requested, &caps, PlannerRule::FirstChange);
        prop_assert_eq!(current.entries().len(), legacy.entries().len());
        for (now, then) in current.entries().iter().zip(legacy.entries()) {
            prop_assert_eq!(now.stage, then.stage);
            prop_assert!(
                now.status.revalidation() >= then.status.revalidation(),
                "{} {:?} under removal-first, {:?} under first-change",
                now.stage,
                now.status,
                then.status
            );
            prop_assert_eq!(now.status.tag(), then.status.tag());
        }
        if !any_removed(&previous, &requested) {
            prop_assert_eq!(&current, &legacy);
        }
    }

    /// A sealed receipt names the rule it was planned under in its version (the legacy version
    /// exactly when both rules give its table), carries the current plan, and replays from its
    /// bytes to the same receipt.
    #[test]
    fn a_receipt_replays_under_the_rule_it_records(states in workflow(), retarget in 0usize..3) {
        let caps = RecalcCapabilities::in_process(RETARGETS[retarget]);
        let previous = identities(&states, false);
        let requested = identities(&states, true);
        let current = RecalcPlan::plan(&previous, &requested, &caps);
        let legacy = RecalcPlan::plan_with_rule(&previous, &requested, &caps, PlannerRule::FirstChange);
        let receipt = RecalcReceiptArtifact::seal(&previous, &requested, &caps, &work(&current))
            .expect("a consistent receipt");
        let version = receipt.meta().version;
        prop_assert_eq!(
            version,
            if current == legacy {
                RECALC_RECEIPT_ARTIFACT_VERSION
            } else {
                RECALC_RECEIPT_REMOVAL_FIRST_VERSION
            }
        );
        prop_assert!(planner_rule_of_version(version).is_some());
        prop_assert_eq!(receipt.plan(), &current);
        let bytes = receipt.to_bytes("standing-laws").expect("encodes");
        let replayed = RecalcReceiptArtifact::from_bytes(&bytes, Some(receipt.receipt_identity()))
            .expect("replays");
        prop_assert_eq!(replayed, receipt);
    }
}
