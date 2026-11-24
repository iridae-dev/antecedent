//! Category laws for the in-crate identity-site tests (formal/identity_sites.toml): every
//! permutation of a small list, a repeated element and a dropped element, checked against the
//! site's classification. `identity` builds the production input from the list and returns the
//! identity the site computes, or `None` when production refuses the input.
//!
//! * `set`: every permutation keeps the identity; a repeat is refused or keeps it; dropping an
//!   element changes it.
//! * `multiset`: every permutation keeps the identity; a repeat changes it.
//! * `ordered`: swapping two different elements changes the identity.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt::Debug;

/// Every permutation of `items` (at most a few elements).
pub(crate) fn permutations<T: Clone>(items: &[T]) -> Vec<Vec<T>> {
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

/// Set laws over distinct `items`.
pub(crate) fn set_laws<T: Clone + Debug, I: PartialEq + Debug>(
    items: &[T],
    identity: impl Fn(&[T]) -> Option<I>,
) {
    let base = identity(items).expect("the base input is admitted");
    for permuted in permutations(items) {
        assert_eq!(
            identity(&permuted).as_ref(),
            Some(&base),
            "reordering {permuted:?} changed a set"
        );
    }
    let mut repeated = items.to_vec();
    repeated.push(items[0].clone());
    if let Some(with_repeat) = identity(&repeated) {
        assert_eq!(with_repeat, base, "a repeated element changed a set");
    }
    if let Some(smaller) = identity(&items[1..]) {
        assert_ne!(smaller, base, "dropping an element left a set unchanged");
    }
}

/// Multiset laws over `items`.
pub(crate) fn multiset_laws<T: Clone + Debug, I: PartialEq + Debug>(
    items: &[T],
    identity: impl Fn(&[T]) -> Option<I>,
) {
    let base = identity(items).expect("the base input is admitted");
    for permuted in permutations(items) {
        assert_eq!(
            identity(&permuted).as_ref(),
            Some(&base),
            "reordering {permuted:?} changed a multiset"
        );
    }
    let mut repeated = items.to_vec();
    repeated.push(items[0].clone());
    let with_repeat = identity(&repeated).expect("a repeat is admitted in a multiset");
    assert_ne!(with_repeat, base, "a repeated element left a multiset unchanged");
}

/// Ordered-list law over `items` whose first two elements differ.
pub(crate) fn ordered_laws<T: Clone + Debug, I: PartialEq + Debug>(
    items: &[T],
    identity: impl Fn(&[T]) -> Option<I>,
) {
    let base = identity(items).expect("the base input is admitted");
    let mut swapped = items.to_vec();
    swapped.swap(0, 1);
    if let Some(reordered) = identity(&swapped) {
        assert_ne!(reordered, base, "reordering an ordered list left it unchanged");
    }
}
