//! Category laws for the identity sites of this crate (`formal/identity_sites.toml`), used by
//! in-crate tests of sites no public API reaches, or whose inputs only the crate's own fixtures
//! build. Each law enumerates every reordering of a small list and the repeat and removal of
//! an element, and takes a per-site adapter `identity` that builds the production input from
//! the list and returns the identity the site computes (`None` when production refuses it).
//!
//! The expected outcome comes from the site's classification, never from an earlier output:
//! a `set` keeps its identity under reordering and repetition and changes it when an element
//! is dropped; a `multiset` keeps it under reordering and changes it on repetition; an
//! `ordered` list changes it when two different elements swap; an injective encoder gives two
//! inputs one identity exactly when their meaning agrees.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt::Debug;

/// Every ordering of `items` (at most 5 elements, so at most 120 orderings).
pub(crate) fn permutations<T: Clone>(items: &[T]) -> Vec<Vec<T>> {
    assert!(items.len() <= 5, "enumerate at most 5 elements");
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }
    let mut out = Vec::new();
    for index in 0..items.len() {
        let mut rest = items.to_vec();
        let head = rest.remove(index);
        for mut tail in permutations(&rest) {
            tail.insert(0, head.clone());
            out.push(tail);
        }
    }
    out
}

fn base<T: Debug, I>(items: &[T], identity: &impl Fn(&[T]) -> Option<I>) -> I {
    identity(items).unwrap_or_else(|| panic!("production refused the base input {items:?}"))
}

fn reorderings_keep<T: Clone + Debug, I: PartialEq + Debug>(
    items: &[T],
    identity: &impl Fn(&[T]) -> Option<I>,
    expected: &I,
) {
    for ordering in permutations(items) {
        let got = identity(&ordering)
            .unwrap_or_else(|| panic!("production refused the reordering {ordering:?}"));
        assert_eq!(&got, expected, "reordering {items:?} as {ordering:?} changed the identity");
    }
}

/// Set laws over distinct `items`: any reordering and a repeated element keep the identity
/// (or the repeat is refused); dropping an element changes it (or the smaller input is refused).
pub(crate) fn assert_set<T: Clone + Debug, I: PartialEq + Debug>(
    items: &[T],
    identity: impl Fn(&[T]) -> Option<I>,
) {
    let expected = base(items, &identity);
    reorderings_keep(items, &identity, &expected);
    if let Some(first) = items.first() {
        let mut repeated = items.to_vec();
        repeated.push(first.clone());
        if let Some(got) = identity(&repeated) {
            assert_eq!(got, expected, "repeating {first:?} changed a set identity");
        }
        if let Some(got) = identity(&items[1..]) {
            assert_ne!(got, expected, "dropping {first:?} left a set identity unchanged");
        }
    }
}

/// Multiset laws over `items`: any reordering keeps the identity; a repeated element changes
/// it (or the repeat is refused).
pub(crate) fn assert_multiset<T: Clone + Debug, I: PartialEq + Debug>(
    items: &[T],
    identity: impl Fn(&[T]) -> Option<I>,
) {
    let expected = base(items, &identity);
    reorderings_keep(items, &identity, &expected);
    if let Some(first) = items.first() {
        let mut repeated = items.to_vec();
        repeated.push(first.clone());
        if let Some(got) = identity(&repeated) {
            assert_ne!(got, expected, "repeating {first:?} left a multiset identity unchanged");
        }
    }
}

/// Ordered-list law over `items`: swapping two adjacent different elements changes the
/// identity (or the reordering is refused).
pub(crate) fn assert_ordered<T: Clone + Debug + PartialEq, I: PartialEq + Debug>(
    items: &[T],
    identity: impl Fn(&[T]) -> Option<I>,
) {
    let expected = base(items, &identity);
    for index in 1..items.len() {
        if items[index - 1] == items[index] {
            continue;
        }
        let mut swapped = items.to_vec();
        swapped.swap(index - 1, index);
        if let Some(got) = identity(&swapped) {
            assert_ne!(got, expected, "swapping positions {} and {index} of {items:?}", index - 1);
        }
    }
}

/// Framing law over `inputs`: two admitted inputs share an identity exactly when their
/// meaning `key` agrees.
pub(crate) fn assert_injective<P: Debug, K: PartialEq, I: PartialEq + Debug>(
    inputs: &[P],
    key: impl Fn(&P) -> K,
    identity: impl Fn(&P) -> Option<I>,
) {
    let identities: Vec<Option<I>> = inputs.iter().map(&identity).collect();
    for (i, left) in inputs.iter().enumerate() {
        for (j, right) in inputs.iter().enumerate().skip(i + 1) {
            if let (Some(a), Some(b)) = (&identities[i], &identities[j]) {
                assert_eq!(key(left) == key(right), a == b, "{left:?} vs {right:?}");
            }
        }
    }
}

/// A set the site accepts only in its canonical (sorted, distinct) spelling: `items` in that
/// spelling is admitted, every other ordering and a repeat are refused, and dropping an element
/// changes the identity.
pub(crate) fn assert_canonical_set<T: Clone + Debug + PartialEq, I: PartialEq + Debug>(
    items: &[T],
    identity: impl Fn(&[T]) -> Option<I>,
) {
    let expected = base(items, &identity);
    for ordering in permutations(items) {
        if ordering.as_slice() != items {
            assert!(identity(&ordering).is_none(), "the spelling {ordering:?} was admitted");
        }
    }
    if let Some(first) = items.first() {
        let mut repeated = items.to_vec();
        repeated.insert(1, first.clone());
        assert!(identity(&repeated).is_none(), "the repeated element {first:?} was admitted");
        if let Some(got) = identity(&items[1..]) {
            assert_ne!(got, expected, "dropping {first:?} left the identity unchanged");
        }
    }
}
