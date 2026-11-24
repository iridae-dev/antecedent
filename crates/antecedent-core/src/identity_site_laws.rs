//! Category laws for the identity sites of `formal/identity_sites.toml`, checked
//! exhaustively over small lists inside the crate (the sites' types are built from the
//! crate's own test fixtures).
//!
//! Each law takes the elements of one collection and an adapter that builds the production
//! value with that collection and returns its canonical digest (`None` when production
//! refuses the input). Every permutation, every single repeat, every single drop and every
//! swap of two different elements is tried:
//!
//! * set: every permutation and every repeat keep the digest; every drop changes it;
//! * multiset: every permutation keeps it; every repeat changes it;
//! * ordered: every swap of two different elements changes it.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::canonical_encode::CanonicalEncode;
use crate::identity::CanonicalWriter;

/// The canonical digest of `value` under a fixed test tag.
pub(crate) fn digest<T: CanonicalEncode>(value: &T) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("identity_site");
    value.encode(&mut writer);
    writer.finish()
}

/// Every permutation of `items` (Heap's algorithm).
pub(crate) fn permutations<T: Clone>(items: &[T]) -> Vec<Vec<T>> {
    fn heap<T: Clone>(k: usize, items: &mut Vec<T>, out: &mut Vec<Vec<T>>) {
        if k <= 1 {
            out.push(items.clone());
            return;
        }
        for i in 0..k {
            heap(k - 1, items, out);
            if k % 2 == 0 {
                items.swap(i, k - 1);
            } else {
                items.swap(0, k - 1);
            }
        }
    }
    let mut work = items.to_vec();
    let mut out = Vec::new();
    heap(work.len(), &mut work, &mut out);
    out
}

fn base<T, I>(items: &[T], identity: &impl Fn(&[T]) -> Option<I>) -> I {
    identity(items).expect("production accepts the base input of an identity-site law")
}

fn repeated<T: Clone>(items: &[T], index: usize) -> Vec<T> {
    let mut out = items.to_vec();
    out.insert(index + 1, items[index].clone());
    out
}

fn dropped<T: Clone>(items: &[T], index: usize) -> Vec<T> {
    let mut out = items.to_vec();
    out.remove(index);
    out
}

/// Set law over distinct `items`.
pub(crate) fn set_law<T: Clone, I: PartialEq + std::fmt::Debug>(
    items: &[T],
    identity: impl Fn(&[T]) -> Option<I>,
) {
    let base = base(items, &identity);
    for permutation in permutations(items) {
        if let Some(digest) = identity(&permutation) {
            assert_eq!(digest, base, "a reordering changed a set");
        }
    }
    for index in 0..items.len() {
        if let Some(digest) = identity(&repeated(items, index)) {
            assert_eq!(digest, base, "a repeated element changed a set");
        }
        if let Some(digest) = identity(&dropped(items, index)) {
            assert_ne!(digest, base, "dropping an element left a set unchanged");
        }
    }
}

/// Multiset law over `items`.
pub(crate) fn multiset_law<T: Clone, I: PartialEq + std::fmt::Debug>(
    items: &[T],
    identity: impl Fn(&[T]) -> Option<I>,
) {
    let base = base(items, &identity);
    for permutation in permutations(items) {
        if let Some(digest) = identity(&permutation) {
            assert_eq!(digest, base, "a reordering changed a multiset");
        }
    }
    for index in 0..items.len() {
        if let Some(digest) = identity(&repeated(items, index)) {
            assert_ne!(digest, base, "a repeated element left a multiset unchanged");
        }
        if let Some(digest) = identity(&dropped(items, index)) {
            assert_ne!(digest, base, "dropping an element left a multiset unchanged");
        }
    }
}

/// Ordered law over `items`: every swap of two different elements changes the digest.
pub(crate) fn ordered_law<T: Clone + PartialEq, I: PartialEq + std::fmt::Debug>(
    items: &[T],
    identity: impl Fn(&[T]) -> Option<I>,
) {
    let base = base(items, &identity);
    for i in 0..items.len() {
        for j in i + 1..items.len() {
            if items[i] == items[j] {
                continue;
            }
            let mut swapped = items.to_vec();
            swapped.swap(i, j);
            if let Some(digest) = identity(&swapped) {
                assert_ne!(digest, base, "reordering an ordered list left it unchanged");
            }
        }
    }
    for index in 0..items.len() {
        if let Some(digest) = identity(&dropped(items, index)) {
            assert_ne!(digest, base, "dropping an element left an ordered list unchanged");
        }
    }
}

/// Injectivity over a family of inputs: two inputs share a digest exactly when their
/// canonical keys agree.
pub(crate) fn injective<P, K: PartialEq + std::fmt::Debug, I: PartialEq>(
    inputs: &[P],
    key: impl Fn(&P) -> K,
    identity: impl Fn(&P) -> Option<I>,
) {
    let digests: Vec<Option<I>> = inputs.iter().map(&identity).collect();
    for (i, left) in inputs.iter().enumerate() {
        for (j, right) in inputs.iter().enumerate().skip(i + 1) {
            if let (Some(a), Some(b)) = (&digests[i], &digests[j]) {
                assert_eq!(
                    key(left) == key(right),
                    a == b,
                    "inputs {i} ({:?}) and {j} ({:?})",
                    key(left),
                    key(right)
                );
            }
        }
    }
}

#[test]
fn the_laws_enumerate_every_permutation() {
    assert_eq!(permutations(&[1, 2, 3, 4]).len(), 24);
    let mut all = permutations(&[1, 2, 3]);
    all.sort_unstable();
    all.dedup();
    assert_eq!(all.len(), 6);
}
