//! Category laws for the identity sites of the compiled surface (formal/identity_sites.toml),
//! shared by the in-crate site tests. The facade's program identities and question digests are
//! private to the crate, so their checks are unit tests here; this is the deterministic
//! counterpart of the proptest harness in formal/adversarial/tests/common/mod.rs (proptest never
//! enters the main workspace lockfile). Each law takes elements and a per-site adapter `identity`
//! that builds the production input from them and returns the identity the site computes
//! (`None` when production refuses the input).
//!
//! The expected outcome comes from the site's classification, never from an earlier output:
//!
//! * `set`: every rotation and the reversal keep the identity; repeating an element keeps it;
//!   dropping an element that occurs once changes it.
//! * `multiset`: every rotation and the reversal keep the identity; repeating an element changes it.
//! * `ordered`: swapping two different elements changes the identity.
//! * framing: two different part tuples never share an identity (`injective`).
//! * reals: `-0.0` and `0.0`, and two NaN payloads, are different inputs (`bit_exact`).
//!
//! The base input of every law must be accepted: a site test chooses inputs production takes,
//! so a refused base is a failure, never a skipped case. A refused variant (a repetition a set
//! refuses before encoding, say) is allowed and skips that one law.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(dead_code, reason = "each site test uses a subset of the laws")]

use std::fmt::Debug;

/// Every rotation of `items` and its reversal: reorderings a set or multiset must not see.
fn reorderings<T: Clone>(items: &[T]) -> Vec<Vec<T>> {
    let mut out = Vec::new();
    for shift in 1..items.len() {
        let mut rotated = items.to_vec();
        rotated.rotate_left(shift);
        out.push(rotated);
    }
    let mut reversed = items.to_vec();
    reversed.reverse();
    out.push(reversed);
    out
}

fn accepted<I>(value: Option<I>, what: &str) -> I {
    value.unwrap_or_else(|| panic!("production refused the {what} input of a site law"))
}

/// Set laws over distinct `items` (at least two).
///
/// # Panics
/// A law fails, or production refuses the base input or a reordering of it.
pub(crate) fn set_laws<T: Clone + Debug + PartialEq, I: PartialEq + Debug>(
    items: &[T],
    identity: impl Fn(&[T]) -> Option<I>,
) {
    assert!(items.len() >= 2, "a set law needs at least two elements");
    let base = accepted(identity(items), "base");
    for reordered in reorderings(items) {
        assert_eq!(base, accepted(identity(&reordered), "reordered"), "a reordering changed a set");
    }
    let mut repeated = items.to_vec();
    repeated.push(items[0].clone());
    if let Some(with_repeat) = identity(&repeated) {
        assert_eq!(base, with_repeat, "a repeated element changed a set");
    }
    if let Some(smaller) = identity(&items[1..]) {
        assert_ne!(base, smaller, "dropping an element left a set unchanged");
    }
}

/// Multiset laws over `items` (at least two, not all equal).
///
/// # Panics
/// A law fails, or production refuses the base input, a reordering or the repetition.
pub(crate) fn multiset_laws<T: Clone + Debug + PartialEq, I: PartialEq + Debug>(
    items: &[T],
    identity: impl Fn(&[T]) -> Option<I>,
) {
    assert!(items.len() >= 2, "a multiset law needs at least two elements");
    let base = accepted(identity(items), "base");
    for reordered in reorderings(items) {
        assert_eq!(
            base,
            accepted(identity(&reordered), "reordered"),
            "a reordering changed a multiset"
        );
    }
    let mut repeated = items.to_vec();
    repeated.push(items[0].clone());
    if let Some(with_repeat) = identity(&repeated) {
        assert_ne!(base, with_repeat, "a repeated element left a multiset unchanged");
    }
}

/// Ordered-list law: swapping any two different elements changes the identity.
///
/// # Panics
/// A law fails, or production refuses the base input.
pub(crate) fn ordered_laws<T: Clone + Debug + PartialEq, I: PartialEq + Debug>(
    items: &[T],
    identity: impl Fn(&[T]) -> Option<I>,
) {
    assert!(items.len() >= 2, "an ordered law needs at least two elements");
    let base = accepted(identity(items), "base");
    for i in 0..items.len() {
        for j in i + 1..items.len() {
            if items[i] == items[j] {
                continue;
            }
            let mut swapped = items.to_vec();
            swapped.swap(i, j);
            if let Some(reordered) = identity(&swapped) {
                assert_ne!(base, reordered, "swapping {i} and {j} left an ordered list unchanged");
            }
        }
    }
}

/// Framing law over every pair of `parts`: two inputs share an identity exactly when their
/// canonical parts (the site's own meaning-preserving normalization) agree.
///
/// # Panics
/// A law fails, or production refuses one of the inputs.
pub(crate) fn injective<P: Debug, K: PartialEq, I: PartialEq + Debug>(
    parts: &[P],
    canonical: impl Fn(&P) -> K,
    identity: impl Fn(&P) -> Option<I>,
) {
    let identities: Vec<I> = parts.iter().map(|p| accepted(identity(p), "framing")).collect();
    for (i, left) in parts.iter().enumerate() {
        for (j, right) in parts.iter().enumerate().skip(i + 1) {
            assert_eq!(
                canonical(left) == canonical(right),
                identities[i] == identities[j],
                "{left:?} vs {right:?}"
            );
        }
    }
}

/// Every identity in `identities` differs from every other: each names a different input.
///
/// # Panics
/// Two identities agree.
pub(crate) fn all_distinct<I: PartialEq + Debug>(identities: &[I]) {
    for (i, left) in identities.iter().enumerate() {
        for (j, right) in identities.iter().enumerate().skip(i + 1) {
            assert_ne!(left, right, "inputs {i} and {j} share an identity");
        }
    }
}

/// Reals that are equal as numbers but differ in bits.
pub(crate) fn bit_distinct_pairs() -> [(f64, f64); 2] {
    [(0.0, -0.0), (f64::NAN, f64::from_bits(f64::NAN.to_bits() ^ 1))]
}

/// Bit-exactness law: for each pair of reals equal as numbers (or both NaN) but different in
/// bits, the identity differs, unless production refuses one of them.
///
/// # Panics
/// Two bit-distinct reals share an identity.
pub(crate) fn bit_exact<I: PartialEq + Debug>(identity: impl Fn(f64) -> Option<I>) {
    for (left, right) in bit_distinct_pairs() {
        if let (Some(a), Some(b)) = (identity(left), identity(right)) {
            assert_ne!(a, b, "{left} and {right} share an identity");
        }
    }
}

/// Text parts that contain the separators unframed encoders join with.
pub(crate) const ADVERSARIAL_TEXT: [(&str, &str); 4] =
    [("ab", "c"), ("a", "bc"), ("a:b", ""), ("a", ":b")];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_laws_accept_a_set_a_multiset_and_a_list() {
        let set = |items: &[u8]| {
            let mut sorted = items.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            Some(sorted)
        };
        let multiset = |items: &[u8]| {
            let mut sorted = items.to_vec();
            sorted.sort_unstable();
            Some(sorted)
        };
        let list = |items: &[u8]| Some(items.to_vec());
        set_laws(&[1, 2, 3], set);
        multiset_laws(&[1, 2, 2], multiset);
        ordered_laws(&[1, 2, 3], list);
        bit_exact(|x| Some(x.to_bits()));
        injective(&ADVERSARIAL_TEXT, |p| *p, |(a, b)| Some((a.len(), format!("{a}{b}"))));
    }

    #[test]
    #[should_panic(expected = "a reordering changed a set")]
    fn the_set_law_rejects_an_ordered_encoding() {
        set_laws(&[1_u8, 2, 3], |items| Some(items.to_vec()));
    }

    #[test]
    #[should_panic(expected = "a repeated element left a multiset unchanged")]
    fn the_multiset_law_rejects_a_deduplicating_encoding() {
        multiset_laws(&[1_u8, 2], |items| {
            let mut sorted = items.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            Some(sorted)
        });
    }

    #[test]
    #[should_panic(expected = "left an ordered list unchanged")]
    fn the_ordered_law_rejects_a_sorting_encoding() {
        ordered_laws(&[2_u8, 1], |items| {
            let mut sorted = items.to_vec();
            sorted.sort_unstable();
            Some(sorted)
        });
    }

    #[test]
    #[should_panic(expected = "share an identity")]
    fn the_bit_law_rejects_a_numeric_encoding() {
        bit_exact(|x| Some(if x.is_nan() { 1 } else { u8::from(x == 0.0) }));
    }
}
