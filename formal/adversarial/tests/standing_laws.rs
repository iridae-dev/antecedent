//! Standing-transition laws over generated inputs, against the production functions the
//! standing-site inventory (`formal/standing_sites.toml`) names.
//!
//! The expected outcomes come from the rules proved in `formal/Antecedent/Standing.lean`
//! and `formal/Antecedent/Revalidation.lean`: standing is never raised without evidence, a
//! changed input lowers what rests on it (a removed one withdraws it) and only a
//! recomputation raises it again, and a composition keeps what any step left unresolved.

use std::sync::Arc;

use antecedent_contract::composition_verification::ObservedEffect;
use antecedent_contract::{DomainStatus, HandoffReceipt, IDENTIFICATION_NOT_CLAIMED};
use antecedent_core::SemanticDigest;
use antecedent_core::recalc::{
    RecalcCapabilities, RecalcPlan, ResumeContext, RetargetSupport, Revalidation, Stage,
    StageIdentities, StageIdentity, StageStatus,
};
use antecedent_io_core::{
    IdentificationSlotWire, SlotSectionWire, SupportSlotWire, claim_domain_statuses_of_section,
};
use proptest::prelude::*;

fn config() -> ProptestConfig {
    ProptestConfig { cases: 256, failure_persistence: None, ..ProptestConfig::default() }
}

// -- recalculation plans --------------------------------------------------------------

const RETARGETS: [RetargetSupport; 3] =
    [RetargetSupport::Supported, RetargetSupport::NotDeclared, RetargetSupport::Incompatible];

/// Per stage: its version before (0 = not declared), whether the request keeps it, and its
/// version after (0 = removed). Several stages change at once, removals among them.
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

fn removed(previous: &StageIdentities, requested: &StageIdentities, stage: Stage) -> bool {
    previous.contains(stage) && !requested.contains(stage)
}

/// The class an observed effect gives a claim graph node: none is still valid, a refusal or
/// a withdrawal withdraws, a recomputation recomputes.
fn effect_class(effect: Option<&ObservedEffect>) -> Revalidation {
    match effect {
        None => Revalidation::StillValid,
        Some(ObservedEffect::Recomputed { .. }) => Revalidation::Recompute,
        Some(ObservedEffect::Withdrawn { .. } | ObservedEffect::Refused { .. }) => {
            Revalidation::Withdrawn
        }
    }
}

// -- claim domains --------------------------------------------------------------------

const STATUSES: [&str; 8] = [
    "nonparametrically_identified",
    "identified_under_parametric_restrictions",
    "identified_under_prior_restrictions",
    "graph_dependent",
    "partially_identified",
    "not_identified",
    "not_certified",
    "an_unknown_spelling",
];
const MASSES: [f64; 4] = [0.0, 0.25, 0.5, 1.0];
const MATRIX: [&str; 5] =
    ["supported", "not_supported", "not_applicable", "unknown", "experimental"];
const EMPIRICAL: [&str; 6] = [
    "supported",
    "failed:overlap.trimming",
    "limited:overlap_policy_bound",
    "unavailable:not_evaluated",
    "not_evaluated",
    "supported:extra",
];

fn identification_slot() -> impl Strategy<Value = IdentificationSlotWire> {
    (0usize..STATUSES.len(), [0usize..4, 0usize..4, 0usize..4, 0usize..4], any::<bool>()).prop_map(
        |(status, [identified, unidentified, unevaluable, incomplete], weighted)| {
            IdentificationSlotWire {
                status: STATUSES[status].to_owned(),
                identified_mass: MASSES[identified],
                unidentified_mass: MASSES[unidentified],
                unevaluable_mass: MASSES[unevaluable],
                incomplete_search_mass: MASSES[incomplete],
                full_mass_scope: true,
                search_capped: incomplete > 0,
                weight_basis: weighted.then(|| "posterior_probability".to_owned()),
            }
        },
    )
}

/// A section that carries a slot, states identification is not claimed, or is unavailable
/// for another reason.
fn identification_section() -> impl Strategy<Value = SlotSectionWire<IdentificationSlotWire>> {
    prop_oneof![
        identification_slot()
            .prop_map(|slot| SlotSectionWire { value: Some(slot), unavailable: None }),
        Just(SlotSectionWire {
            value: None,
            unavailable: Some(IDENTIFICATION_NOT_CLAIMED.to_owned())
        }),
        Just(SlotSectionWire { value: None, unavailable: Some("not_computed".to_owned()) }),
    ]
}

fn support_slot() -> impl Strategy<Value = Option<SupportSlotWire>> {
    prop::option::of((0usize..MATRIX.len(), 0usize..EMPIRICAL.len()).prop_map(|(m, e)| {
        SupportSlotWire {
            matrix_status: MATRIX[m].to_owned(),
            matrix_coordinate: None,
            empirical: EMPIRICAL[e].to_owned(),
        }
    }))
}

// -- handoff receipts -----------------------------------------------------------------

const FIELDS: [&str; 6] =
    ["claim.envelope", "identities", "reasoning", "domains", "value", "bytes"];

fn fields() -> impl Strategy<Value = Vec<Arc<str>>> {
    prop::collection::vec(prop::sample::select(FIELDS.to_vec()), 0..5)
        .prop_map(|items| items.into_iter().map(Arc::from).collect())
}

/// A chain of hops about one claim path: each hop's input is the previous hop's output.
fn hops() -> impl Strategy<Value = Vec<HandoffReceipt>> {
    prop::collection::vec((fields(), fields(), fields(), any::<bool>()), 1..6).prop_map(|specs| {
        specs
            .into_iter()
            .enumerate()
            .map(|(index, (retained, omitted, unresolved, contiguous))| {
                let input = u8::try_from(index).expect("few hops") + 1;
                let output = if contiguous { input + 1 } else { input + 100 };
                HandoffReceipt::new(
                    SemanticDigest::from_bytes([input; 32]),
                    Some(SemanticDigest::from_bytes([output; 32])),
                    "consumer",
                    "rule",
                    retained,
                    omitted,
                    unresolved,
                    Vec::<Arc<str>>::new(),
                )
            })
            .collect()
    })
}

fn has(list: &[Arc<str>], field: &str) -> bool {
    list.iter().any(|item| &**item == field)
}

proptest! {
    #![proptest_config(config())]

    /// In-process recalculation never leaves a stage in a better class than any stage it
    /// rests on, and a stage resting on an input that is gone is withdrawn, whatever else
    /// changed beside it: a withdrawal is never read as a recomputation.
    #[test]
    fn recalc_dependents_never_outrank_their_dependencies(
        states in workflow(),
        retarget in 0usize..3,
    ) {
        let retarget = RETARGETS[retarget];
        let previous = identities(&states, false);
        let requested = identities(&states, true);
        let plan =
            RecalcPlan::plan(&previous, &requested, &RecalcCapabilities::in_process(retarget));
        for entry in plan.entries() {
            let class = entry.status.revalidation();
            for dep in entry.stage.dependencies(retarget) {
                if removed(&previous, &requested, dep) {
                    prop_assert_eq!(
                        class,
                        Revalidation::Withdrawn,
                        "{} rests on removed {}: {:?}",
                        entry.stage,
                        dep,
                        entry.status
                    );
                }
                if let Some(upstream) = plan.revalidation(dep) {
                    prop_assert!(
                        class >= upstream,
                        "{} is {:?} but {} is {:?}",
                        entry.stage,
                        class,
                        dep,
                        upstream
                    );
                }
            }
        }
    }

    /// The effect a recalculation stage reports into a claim graph is exactly its class:
    /// a recomputation that rests on a removed input is a withdrawal, never a numerical
    /// effect, and only a reuse is no effect.
    #[test]
    fn stage_effect_is_its_class(
        states in workflow(),
        retarget in 0usize..3,
        fresh in any::<bool>(),
        resume in (any::<bool>(), any::<bool>(), any::<bool>(), any::<bool>(), any::<bool>()),
    ) {
        let retarget = RETARGETS[retarget];
        let previous = identities(&states, false);
        let requested = identities(&states, true);
        let caps = if fresh {
            let (portable_fit, portable_scores, scores_snapshot_bound, supplied_data, supplied_provider) = resume;
            RecalcCapabilities::fresh_process(retarget, ResumeContext {
                portable_fit,
                portable_scores,
                scores_snapshot_bound,
                supplied_data,
                supplied_provider,
            })
        } else {
            RecalcCapabilities::in_process(retarget)
        };
        let plan = RecalcPlan::plan(&previous, &requested, &caps);
        for entry in plan.entries() {
            let effect = ObservedEffect::from_stage_status(&entry.status);
            prop_assert_eq!(
                effect_class(effect.as_ref()),
                entry.status.revalidation(),
                "{} {:?} -> {:?}",
                entry.stage,
                entry.status,
                effect
            );
            prop_assert_eq!(effect.is_none(), matches!(entry.status, StageStatus::Reused { .. }));
        }
    }

    /// The domains a producer and an independent consumer derive from a claim's sections
    /// assert nothing the sections do not establish: identified (and evaluated) only for a
    /// nonparametric or parametric status with no mass outside identification, supported
    /// only when the matrix and the empirical check both support it, not claimed only when
    /// the section says so.
    #[test]
    fn claim_domains_never_raise_without_evidence(
        identification in identification_section(),
        support in support_slot(),
    ) {
        let domains = claim_domain_statuses_of_section(support.as_ref(), &identification);
        let not_claimed = identification.value.is_none()
            && identification.unavailable.as_deref() == Some(IDENTIFICATION_NOT_CLAIMED);
        let established = identification.value.as_ref().is_some_and(|slot| {
            matches!(
                slot.status.as_str(),
                "nonparametrically_identified" | "identified_under_parametric_restrictions"
            ) && slot.unidentified_mass == 0.0
                && slot.unevaluable_mass == 0.0
                && slot.incomplete_search_mass == 0.0
        });
        prop_assert_eq!(domains.identification == DomainStatus::Identified, established);
        prop_assert_eq!(domains.identification == DomainStatus::NotClaimed, not_claimed);
        if domains.evaluated == DomainStatus::Evaluated {
            prop_assert!(established || not_claimed);
        }
        let stated_support = support
            .as_ref()
            .is_some_and(|slot| slot.matrix_status == "supported" && slot.empirical == "supported");
        prop_assert_eq!(domains.support == DomainStatus::Supported, stated_support);
        for status in [domains.identification, domains.support, domains.evaluated] {
            prop_assert!(!matches!(status, DomainStatus::Supported) || stated_support);
        }
    }

    /// Along any chain of handoff hops, nothing any hop omitted is ever retained, every
    /// unresolved item of any hop stays unresolved, and the chain is equivalent to its input
    /// claim only if no hop lost anything.
    #[test]
    fn handoff_chain_of_any_length_never_restores_a_loss(chain in hops()) {
        let folded = chain[1..].iter().fold(chain[0].clone(), |acc, next| acc.chain(next));
        for field in FIELDS {
            let omitted_anywhere = chain.iter().any(|hop| has(&hop.omitted, field));
            prop_assert_eq!(has(&folded.omitted, field), omitted_anywhere);
            if omitted_anywhere {
                prop_assert!(!has(&folded.retained, field), "{} restored", field);
            }
            if chain.iter().any(|hop| has(&hop.unresolved, field)) {
                prop_assert!(has(&folded.unresolved, field), "{} resolved by chaining", field);
            }
        }
        if folded.equivalent_claim() {
            prop_assert!(chain.iter().all(|hop| hop.omitted.is_empty() && hop.unresolved.is_empty()));
        }
    }
}
