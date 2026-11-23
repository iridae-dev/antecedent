//! `TransportIdentity::digest` is insensitive to the order of source populations and of
//! assumptions, and sensitive to every other change of meaning.
//!
//! Positive cases permute either list. Negative cases add, remove or replace an element,
//! change the target population or the selection diagram, or move an element from one list
//! to the other. Each negative case first checks that the mutated identity really differs as
//! a multiset, so a digest difference is owed rather than assumed.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::evidence_ledger::TransportIdentity;
use antecedent_core::SemanticDigest;
use proptest::prelude::*;

fn config() -> ProptestConfig {
    ProptestConfig { cases: 256, failure_persistence: None, ..ProptestConfig::default() }
}

fn texts() -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec("[a-z]{0,4}", 0..6)
}

fn identity() -> impl Strategy<Value = TransportIdentity> {
    (texts(), "[a-z]{0,4}", any::<[u8; 32]>(), texts()).prop_map(
        |(source_populations, target_population, diagram, assumptions)| TransportIdentity {
            source_populations,
            target_population,
            selection_diagram: SemanticDigest::from_bytes(diagram),
            assumptions,
        },
    )
}

/// A list together with one of its permutations.
fn permuted() -> impl Strategy<Value = (Vec<String>, Vec<String>)> {
    texts().prop_flat_map(|list| (Just(list.clone()), Just(list).prop_shuffle()))
}

type Canonical = (Vec<String>, String, [u8; 32], Vec<String>);

/// The identity as a multiset pair, independent of list order.
fn canonical(identity: &TransportIdentity) -> Canonical {
    let mut populations = identity.source_populations.clone();
    populations.sort_unstable();
    let mut assumptions = identity.assumptions.clone();
    assumptions.sort_unstable();
    (
        populations,
        identity.target_population.clone(),
        *identity.selection_diagram.as_bytes(),
        assumptions,
    )
}

fn flipped(bytes: [u8; 32], index: usize, mask: u8) -> [u8; 32] {
    let mut out = bytes;
    out[index % 32] ^= mask;
    out
}

#[derive(Clone, Debug)]
enum Mutation {
    AddPopulation(String),
    RemovePopulation(usize),
    ReplacePopulation(usize),
    AddAssumption(String),
    RemoveAssumption(usize),
    ReplaceAssumption(usize),
    Target,
    Diagram(usize, u8),
}

fn mutation() -> impl Strategy<Value = Mutation> {
    prop_oneof![
        "[a-z]{0,4}".prop_map(Mutation::AddPopulation),
        any::<usize>().prop_map(Mutation::RemovePopulation),
        any::<usize>().prop_map(Mutation::ReplacePopulation),
        "[a-z]{0,4}".prop_map(Mutation::AddAssumption),
        any::<usize>().prop_map(Mutation::RemoveAssumption),
        any::<usize>().prop_map(Mutation::ReplaceAssumption),
        Just(Mutation::Target),
        (any::<usize>(), 1..=255u8).prop_map(|(index, mask)| Mutation::Diagram(index, mask)),
    ]
}

/// Apply `mutation`; `None` when it does not apply (removing from an empty list).
fn apply(identity: &TransportIdentity, mutation: &Mutation) -> Option<TransportIdentity> {
    let mut out = identity.clone();
    match mutation {
        Mutation::AddPopulation(text) => out.source_populations.push(text.clone()),
        Mutation::AddAssumption(text) => out.assumptions.push(text.clone()),
        Mutation::RemovePopulation(index) => {
            if out.source_populations.is_empty() {
                return None;
            }
            out.source_populations.remove(index % identity.source_populations.len());
        }
        Mutation::RemoveAssumption(index) => {
            if out.assumptions.is_empty() {
                return None;
            }
            out.assumptions.remove(index % identity.assumptions.len());
        }
        Mutation::ReplacePopulation(index) => {
            if out.source_populations.is_empty() {
                return None;
            }
            // `!` is outside the generated alphabet, so the new element is a different string
            // and the multiset loses the old element without regaining it.
            let slot = index % identity.source_populations.len();
            out.source_populations[slot].push('!');
        }
        Mutation::ReplaceAssumption(index) => {
            if out.assumptions.is_empty() {
                return None;
            }
            let slot = index % identity.assumptions.len();
            out.assumptions[slot].push('!');
        }
        Mutation::Target => out.target_population.push('!'),
        Mutation::Diagram(index, mask) => {
            out.selection_diagram = SemanticDigest::from_bytes(flipped(
                *identity.selection_diagram.as_bytes(),
                *index,
                *mask,
            ));
        }
    }
    Some(out)
}

proptest! {
    #![proptest_config(config())]

    /// Any permutation of the source populations and of the assumptions keeps the digest.
    #[test]
    fn identity_permutation_keeps_digest(
        (populations, populations_shuffled) in permuted(),
        (assumptions, assumptions_shuffled) in permuted(),
        target in "[a-z]{0,4}",
        diagram in any::<[u8; 32]>(),
    ) {
        let original = TransportIdentity {
            source_populations: populations,
            target_population: target.clone(),
            selection_diagram: SemanticDigest::from_bytes(diagram),
            assumptions,
        };
        let reordered = TransportIdentity {
            source_populations: populations_shuffled,
            target_population: target,
            selection_diagram: SemanticDigest::from_bytes(diagram),
            assumptions: assumptions_shuffled,
        };
        prop_assert_eq!(original.digest(), reordered.digest());
    }

    /// Adding, removing or replacing a list element, or changing the target population or the
    /// selection diagram, changes the digest.
    #[test]
    fn identity_relevant_change_breaks_digest(base in identity(), change in mutation()) {
        let mutated = apply(&base, &change);
        prop_assume!(mutated.is_some());
        let mutated = mutated.unwrap();
        prop_assert_ne!(canonical(&base), canonical(&mutated), "the mutation must change the multisets");
        prop_assert_ne!(base.digest(), mutated.digest());
    }

    /// Moving an element between the two lists changes the digest: the lists cannot trade
    /// elements.
    #[test]
    fn identity_lists_cannot_trade_elements(
        base in identity(),
        index in any::<usize>(),
        toward_assumptions in any::<bool>(),
    ) {
        let mut moved = base.clone();
        let (from, to) = if toward_assumptions {
            (&mut moved.source_populations, &mut moved.assumptions)
        } else {
            (&mut moved.assumptions, &mut moved.source_populations)
        };
        prop_assume!(!from.is_empty());
        let element = from.remove(index % from.len());
        to.push(element);
        prop_assert_ne!(canonical(&base), canonical(&moved), "the move must change the multisets");
        prop_assert_ne!(base.digest(), moved.digest());
    }
}
