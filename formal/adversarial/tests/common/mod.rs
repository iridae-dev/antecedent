//! Category laws for identity sites (formal/identity_sites.toml), shared by the per-crate
//! site tests. Each law takes generated parts and a per-site adapter `identity` that builds
//! the production input from the parts and returns the identity the site computes (`None`
//! when production refuses the input, which skips the case).
//!
//! The expected outcome comes from the site's classification, never from an earlier output:
//!
//! * `set`: reordering keeps the identity; repeating an element keeps it; dropping an element
//!   that occurs once changes it.
//! * `multiset`: reordering keeps the identity; repeating an element changes it.
//! * `ordered`: swapping two different elements changes the identity.
//! * framing: two different part tuples never share an identity (`injective`).
//! * reals: `-0.0` and `0.0`, and two NaN payloads, are different inputs (`bit_exact`).
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(dead_code, reason = "each site test binary uses a subset of the laws")]

use std::fmt::Debug;

use proptest::prelude::*;
use proptest::test_runner::TestCaseResult;

/// The proptest configuration of every site test.
#[must_use]
pub fn config() -> ProptestConfig {
    ProptestConfig { cases: 128, failure_persistence: None, ..ProptestConfig::default() }
}

/// A list with one of its permutations.
pub fn permuted<T: Clone + Debug + 'static>(
    element: impl Strategy<Value = T> + 'static,
    len: std::ops::Range<usize>,
) -> impl Strategy<Value = (Vec<T>, Vec<T>)> {
    prop::collection::vec(element, len)
        .prop_flat_map(|list| (Just(list.clone()), Just(list).prop_shuffle()))
}

/// Distinct elements (a set as a list in generated order) with one of their permutations.
pub fn distinct_permuted<T: Clone + Debug + Ord + 'static>(
    element: impl Strategy<Value = T> + 'static,
    len: std::ops::Range<usize>,
) -> impl Strategy<Value = (Vec<T>, Vec<T>)> {
    prop::collection::btree_set(element, len).prop_flat_map(|set| {
        let list: Vec<T> = set.into_iter().collect();
        (Just(list.clone()), Just(list).prop_shuffle())
    })
}

fn skip_if_refused<I>(value: Option<I>) -> Result<I, TestCaseError> {
    value.ok_or_else(|| TestCaseError::reject("production refused the generated input"))
}

/// Set laws over a list of distinct elements `items` and its permutation `shuffled`.
///
/// # Errors
/// A failed law; a rejection when production refuses the base input.
pub fn set_laws<T: Clone + Debug + PartialEq, I: PartialEq + Debug>(
    items: &[T],
    shuffled: &[T],
    identity: impl Fn(&[T]) -> Option<I>,
) -> TestCaseResult {
    let base = skip_if_refused(identity(items))?;
    prop_assert_eq!(&base, &skip_if_refused(identity(shuffled))?, "a reordering changed a set");
    if let Some(first) = items.first() {
        let mut repeated = items.to_vec();
        repeated.push(first.clone());
        if let Some(with_repeat) = identity(&repeated) {
            prop_assert_eq!(&base, &with_repeat, "a repeated element changed a set");
        }
        let dropped: Vec<T> = items[1..].to_vec();
        if let Some(smaller) = identity(&dropped) {
            prop_assert_ne!(&base, &smaller, "dropping an element left a set unchanged");
        }
    }
    Ok(())
}

/// Multiset laws over a list `items` and its permutation `shuffled`.
///
/// # Errors
/// A failed law; a rejection when production refuses the base input.
pub fn multiset_laws<T: Clone + Debug + PartialEq, I: PartialEq + Debug>(
    items: &[T],
    shuffled: &[T],
    identity: impl Fn(&[T]) -> Option<I>,
) -> TestCaseResult {
    let base = skip_if_refused(identity(items))?;
    prop_assert_eq!(
        &base,
        &skip_if_refused(identity(shuffled))?,
        "a reordering changed a multiset"
    );
    if let Some(first) = items.first() {
        let mut repeated = items.to_vec();
        repeated.push(first.clone());
        if let Some(with_repeat) = identity(&repeated) {
            prop_assert_ne!(&base, &with_repeat, "a repeated element left a multiset unchanged");
        }
    }
    Ok(())
}

/// Ordered-list law: swapping the elements at `i` and `j` (taken modulo the length), when
/// they differ, changes the identity.
///
/// # Errors
/// A failed law; a rejection when production refuses the base input.
pub fn ordered_laws<T: Clone + Debug + PartialEq, I: PartialEq + Debug>(
    items: &[T],
    i: usize,
    j: usize,
    identity: impl Fn(&[T]) -> Option<I>,
) -> TestCaseResult {
    let base = skip_if_refused(identity(items))?;
    if items.len() < 2 {
        return Ok(());
    }
    let (i, j) = (i % items.len(), j % items.len());
    if items[i] == items[j] {
        return Ok(());
    }
    let mut swapped = items.to_vec();
    swapped.swap(i, j);
    if let Some(reordered) = identity(&swapped) {
        prop_assert_ne!(&base, &reordered, "reordering an ordered list left it unchanged");
    }
    Ok(())
}

/// Framing law: two inputs whose parts differ (after `canonical`, the site's own
/// meaning-preserving normalization of the parts) never share an identity, and two inputs
/// whose canonical parts agree share it.
///
/// # Errors
/// A failed law.
pub fn injective<P: Debug, K: PartialEq + Debug, I: PartialEq + Debug>(
    left: &P,
    right: &P,
    canonical: impl Fn(&P) -> K,
    identity: impl Fn(&P) -> Option<I>,
) -> TestCaseResult {
    let (Some(a), Some(b)) = (identity(left), identity(right)) else {
        return Err(TestCaseError::reject("production refused a generated input"));
    };
    prop_assert_eq!(canonical(left) == canonical(right), a == b, "{:?} vs {:?}", left, right);
    Ok(())
}

/// Reals that are equal as numbers but differ in bits.
#[must_use]
pub fn bit_distinct_pairs() -> Vec<(f64, f64)> {
    vec![(0.0, -0.0), (f64::NAN, f64::from_bits(f64::NAN.to_bits() ^ 1))]
}

/// Bit-exactness law: for each pair of reals equal as numbers (or both NaN) but different in
/// bits, the identity differs, unless production refuses one of them.
///
/// # Errors
/// A failed law.
pub fn bit_exact<I: PartialEq + Debug>(identity: impl Fn(f64) -> Option<I>) -> TestCaseResult {
    for (left, right) in bit_distinct_pairs() {
        if let (Some(a), Some(b)) = (identity(left), identity(right)) {
            prop_assert_ne!(a, b, "{} and {} share an identity", left, right);
        }
    }
    Ok(())
}

/// Text parts that contain the separators unframed encoders join with.
pub fn adversarial_text() -> impl Strategy<Value = String> {
    "[a-c:;,|= ]{0,4}"
}
