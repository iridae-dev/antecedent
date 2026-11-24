//! Generated workflows, composition graphs and handoff chains against the
//! production standing transitions: `RecalcPlan::plan`, `revalidate` and
//! `HandoffReceipt::chain`.
//!
//! The expected outcomes come from the rules proved in
//! `formal/Antecedent/Revalidation.lean` and `formal/Antecedent/Standing.lean`:
//! a result keeps its standing only when nothing it rests on changed, a moved
//! provider trust standing never leaves a claim still valid, and a handoff chain
//! never restores what a hop lost.

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent_contract::HandoffReceipt;
use antecedent_contract::composition_verification::{
    CompositionGraph, CompositionNode, DependencyKind, NodeKind, PriorAuthority, TrustStanding,
    revalidate,
};
use antecedent_core::SemanticDigest;
use antecedent_core::recalc::{
    ChangedDependency, RecalcCapabilities, RecalcPlan, ResumeContext, RetargetSupport,
    Revalidation, Stage, StageIdentities, StageIdentity, StageStatus,
};
use proptest::prelude::*;

fn config() -> ProptestConfig {
    ProptestConfig { cases: 256, failure_persistence: None, ..ProptestConfig::default() }
}

// -- recalculation plans --------------------------------------------------------------

const RETARGETS: [RetargetSupport; 3] =
    [RetargetSupport::Supported, RetargetSupport::NotDeclared, RetargetSupport::Incompatible];

/// Per stage: its version before (0 = not declared) and whether the request keeps it.
fn workflow() -> impl Strategy<Value = Vec<(u8, bool, u8)>> {
    prop::collection::vec((0u8..3, prop::bool::weighted(0.8), 0u8..3), Stage::all().len())
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

fn resume() -> impl Strategy<Value = Option<ResumeContext>> {
    prop::option::of(
        (any::<bool>(), any::<bool>(), any::<bool>(), any::<bool>(), any::<bool>()).prop_map(
            |(
                portable_fit,
                portable_scores,
                scores_snapshot_bound,
                supplied_data,
                supplied_provider,
            )| {
                ResumeContext {
                    portable_fit,
                    portable_scores,
                    scores_snapshot_bound,
                    supplied_data,
                    supplied_provider,
                }
            },
        ),
    )
}

// -- composition graphs ---------------------------------------------------------------

const TRUSTS: [TrustStanding; 4] = [
    TrustStanding::NotApplicable,
    TrustStanding::Held,
    TrustStanding::Withdrawn,
    TrustStanding::Undeclared,
];

const DEPENDENCIES: [DependencyKind; 6] = [
    DependencyKind::Data,
    DependencyKind::Evidence,
    DependencyKind::ModelVersion,
    DependencyKind::Support,
    DependencyKind::ProviderTrust,
    DependencyKind::StructuralDerivation,
];

/// Per node: identity byte and trust index before and now, and the kind of its
/// dependency on the node before it.
type NodeSpec = (u8, usize, u8, usize, usize);

fn nodes() -> impl Strategy<Value = Vec<NodeSpec>> {
    prop::collection::vec((1u8..3, 0usize..4, 1u8..3, 0usize..4, 0usize..6), 1..6)
}

fn chain_graph(spec: &[NodeSpec], now: bool) -> CompositionGraph {
    let mut graph = CompositionGraph::new();
    for (index, &(id_before, trust_before, id_now, trust_now, _)) in spec.iter().enumerate() {
        let (id, trust) = if now { (id_now, trust_now) } else { (id_before, trust_before) };
        let node = CompositionNode::new(
            format!("n{index}"),
            NodeKind::Bridge,
            SemanticDigest::from_bytes([id; 32]),
        )
        .with_trust(TRUSTS[trust]);
        graph.add_node(node).expect("a well-formed node");
    }
    for (index, &(.., kind)) in spec.iter().enumerate().skip(1) {
        graph
            .add_edge(&format!("n{index}"), &format!("n{}", index - 1), DEPENDENCIES[kind])
            .expect("an acyclic edge");
    }
    graph
}

// -- handoff receipts -----------------------------------------------------------------

const FIELDS: [&str; 6] =
    ["claim.envelope", "identities", "reasoning", "domains", "value", "bytes"];

fn fields() -> impl Strategy<Value = Vec<Arc<str>>> {
    prop::collection::vec(prop::sample::select(FIELDS.to_vec()), 0..5)
        .prop_map(|items| items.into_iter().map(Arc::from).collect())
}

fn receipt() -> impl Strategy<Value = HandoffReceipt> {
    (1u8..4, prop::option::of(1u8..4), fields(), fields(), fields(), fields()).prop_map(
        |(input, output, retained, omitted, unresolved, unavailable)| {
            HandoffReceipt::new(
                SemanticDigest::from_bytes([input; 32]),
                output.map(|o| SemanticDigest::from_bytes([o; 32])),
                "consumer",
                "rule",
                retained,
                omitted,
                unresolved,
                unavailable,
            )
        },
    )
}

fn has(list: &[Arc<str>], field: &str) -> bool {
    list.iter().any(|item| &**item == field)
}

proptest! {
    #![proptest_config(config())]

    /// A stage is reused (still valid) only when its effective identity is unchanged, and
    /// recomputed for a change only when it changed: standing is lowered on a changed input
    /// and kept only without one.
    #[test]
    fn recalc_reuse_only_without_a_change(
        states in workflow(),
        retarget in 0usize..3,
        resume in resume(),
    ) {
        let retarget = RETARGETS[retarget];
        let previous = identities(&states, false);
        let requested = identities(&states, true);
        let caps = match resume {
            Some(resume) => RecalcCapabilities::fresh_process(retarget, resume),
            None => RecalcCapabilities::in_process(retarget),
        };
        let plan = RecalcPlan::plan(&previous, &requested, &caps);
        let before = previous.effective(retarget);
        let after = requested.effective(retarget);
        for entry in plan.entries() {
            let same = before.get(&entry.stage) == after.get(&entry.stage);
            let reused = matches!(entry.status, StageStatus::Reused { .. });
            prop_assert_eq!(
                entry.status.revalidation() == Revalidation::StillValid,
                reused,
                "{} {:?}",
                entry.stage,
                entry.status
            );
            if reused {
                prop_assert!(same, "{} reused but its identity changed", entry.stage);
            }
            if let StageStatus::Recomputed { because } = &entry.status
                && *because != ChangedDependency::FreshProcess
            {
                prop_assert!(!same, "{} recomputed but its identity is unchanged", entry.stage);
            }
        }
    }

    /// Frozen portable scores are never reused in a fresh process over a data snapshot they
    /// were frozen without, when that snapshot's data was not supplied.
    #[test]
    fn recalc_fresh_scores_never_reused_over_a_new_snapshot(
        versions in prop::collection::vec(1u8..3, Stage::all().len()),
        retarget in 0usize..3,
        portable_fit in any::<bool>(),
        scores_snapshot_bound in any::<bool>(),
        supplied_provider in any::<bool>(),
    ) {
        let retarget = RETARGETS[retarget];
        let mut previous = StageIdentities::new();
        for (stage, version) in Stage::all().into_iter().zip(versions) {
            previous.set(stage, StageIdentity::of(&stage.label(), &[&[version][..]]));
        }
        let requested = previous.clone();
        previous.remove(Stage::DataSnapshot);
        let resume = ResumeContext {
            portable_fit,
            portable_scores: true,
            scores_snapshot_bound,
            supplied_data: false,
            supplied_provider,
        };
        let plan = RecalcPlan::plan(
            &previous,
            &requested,
            &RecalcCapabilities::fresh_process(retarget, resume),
        );
        prop_assert!(
            !matches!(plan.status(Stage::ScoreArtifact), Some(StageStatus::Reused { .. })),
            "{:?}",
            plan.status(Stage::ScoreArtifact)
        );
    }

    /// An unchanged in-process workflow reuses every stage it can serve.
    #[test]
    fn recalc_unchanged_workflow_keeps_standing(states in workflow(), retarget in 0usize..3) {
        let retarget = RETARGETS[retarget];
        let same = identities(&states, false);
        let plan = RecalcPlan::plan(&same, &same, &RecalcCapabilities::in_process(retarget));
        for entry in plan.entries() {
            prop_assert!(
                matches!(entry.status, StageStatus::Reused { .. }),
                "{} {:?}",
                entry.stage,
                entry.status
            );
        }
    }

    /// In a composition graph, a claim whose provider trust moved is never still valid,
    /// a still-valid claim and its dependencies are unchanged, only a still-valid claim
    /// keeps its earlier authority, and nothing asks for recomputation without a change.
    #[test]
    fn revalidation_raises_standing_only_by_recomputation(spec in nodes()) {
        let previous = chain_graph(&spec, false);
        let current = chain_graph(&spec, true);
        let result = revalidate(&previous, &current, &BTreeMap::new()).expect("a valid graph");
        let unchanged = revalidate(&current, &current, &BTreeMap::new()).expect("a valid graph");
        let mut upstream_valid = true;
        for (index, &(id_before, trust_before, id_now, trust_now, _)) in spec.iter().enumerate() {
            let key = format!("n{index}");
            let claim = result.claim(&key).expect("every node is classified");
            prop_assert_eq!(
                claim.authority == PriorAuthority::Retained,
                claim.class == Revalidation::StillValid
            );
            if trust_before != trust_now {
                prop_assert_ne!(claim.class, Revalidation::StillValid, "{}", key);
            }
            if claim.class == Revalidation::StillValid {
                prop_assert!(id_before == id_now && trust_before == trust_now, "{}", key);
                prop_assert!(upstream_valid, "{} still valid on a changed dependency", key);
            }
            upstream_valid = claim.class == Revalidation::StillValid;
            let same = unchanged.claim(&key).expect("every node is classified");
            prop_assert!(same.numerical.is_empty(), "{} recomputes without a change", key);
        }
    }

    /// A handoff chain omits what either hop omitted, retains nothing omitted and nothing
    /// no hop declared, keeps every unresolved item, and is equivalent to its input claim
    /// only when both hops are loss-free and describe one path back to it.
    #[test]
    fn handoff_chain_never_restores_a_loss(first in receipt(), next in receipt()) {
        let chain = first.chain(&next);
        for field in FIELDS {
            prop_assert_eq!(
                has(&chain.omitted, field),
                has(&first.omitted, field) || has(&next.omitted, field)
            );
            if has(&chain.retained, field) {
                prop_assert!(!has(&chain.omitted, field), "{} retained and omitted", field);
                prop_assert!(has(&first.retained, field) || has(&next.retained, field));
            }
            if has(&first.unresolved, field) || has(&next.unresolved, field) {
                prop_assert!(has(&chain.unresolved, field));
            }
        }
        if first.output != Some(next.input) {
            prop_assert!(chain.retained.is_empty());
            prop_assert!(has(&chain.unresolved, "chain_discontinuity"));
            prop_assert!(!chain.equivalent_claim());
        }
        if chain.equivalent_claim() {
            prop_assert!(first.omitted.is_empty() && next.omitted.is_empty());
            prop_assert!(first.unresolved.is_empty() && next.unresolved.is_empty());
            prop_assert_eq!(first.output, Some(next.input));
            prop_assert_eq!(next.output, Some(first.input));
        }
    }
}
