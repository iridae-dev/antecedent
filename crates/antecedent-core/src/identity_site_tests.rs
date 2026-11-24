//! Identity sites of the shared canonical encoders and the small value types
//! (`formal/identity_sites.toml`, `core.canonical_encode.*`, `core.callback.*`,
//! `core.intervention.*`, `core.value.*`, `core.value_expr.*`), checked with the category
//! laws of [`crate::identity_site_laws`].
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::callback::CallbackOutput;
use crate::canonical_encode::{encode_list, encode_set, encode_variable_set, encode_variables};
use crate::identity::CanonicalWriter;
use crate::identity_site_laws::{digest, injective, multiset_law, ordered_law, set_law};
use crate::ids::{DynamicRuleId, VariableId};
use crate::intervention::TemporalPolicy;
use crate::value::Value;
use crate::value_expr::PolicyReads;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn trio() -> [VariableId; 3] {
    [v(5), v(6), v(7)]
}

fn written(write: impl FnOnce(&mut CanonicalWriter)) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("identity_site");
    write(&mut writer);
    writer.finish()
}

fn values() -> [Value; 3] {
    [Value::Int64(1), Value::Float64(0.5), Value::Label(Arc::from("a"))]
}

#[test]
fn site_encode_variables_is_ordered() {
    ordered_law(&trio(), |items| Some(written(|w| encode_variables(w, items))));
}

#[test]
fn site_encode_list_is_ordered() {
    ordered_law(&values(), |items| Some(written(|w| encode_list(w, items))));
}

#[test]
fn site_encode_variable_set_is_a_multiset() {
    multiset_law(&trio(), |items| Some(written(|w| encode_variable_set(w, items))));
}

#[test]
fn site_encode_set_is_a_multiset() {
    multiset_law(&values(), |items| Some(written(|w| encode_set(w, items))));
}

/// Two adjacent framed collections cannot trade elements: an element moved from the first to
/// the second changes the bytes.
#[test]
fn site_encode_set_framing() {
    let pairs: Vec<(Vec<u32>, Vec<u32>)> = vec![
        (vec![5, 6], vec![7]),
        (vec![5], vec![6, 7]),
        (vec![], vec![5, 6, 7]),
        (vec![5, 6, 7], vec![]),
        (vec![6, 5], vec![7]),
    ];
    injective(
        &pairs,
        |(left, right)| {
            let (mut left, mut right) = (left.clone(), right.clone());
            left.sort_unstable();
            right.sort_unstable();
            (left, right)
        },
        |(left, right)| {
            let ids = |raw: &[u32]| raw.iter().map(|&r| v(r)).collect::<Vec<_>>();
            Some(written(|w| {
                encode_variable_set(w, &ids(left));
                encode_variable_set(w, &ids(right));
            }))
        },
    );
}

/// Levels are a set: sorted with `total_cmp` and deduplicated. Reals are bitwise in every
/// canonical encoding, so `-0.0` is its own level spelling (a false distinction against `0.0`,
/// never a collision): `[0.0, -0.0]` canonicalizes to `[-0.0]` in either listed order.
#[test]
fn site_callback_output_levels() {
    set_law(&[0.0, 1.0, 2.5], |items| Some(digest(&CallbackOutput::levels(items.iter().copied()))));
    assert_eq!(
        digest(&CallbackOutput::levels([0.0, -0.0])),
        digest(&CallbackOutput::levels([-0.0, 0.0]))
    );
}

/// A dynamic schedule's active offsets are a set (sorted and deduplicated by the
/// constructor, which `is_active_at`'s binary search relies on).
#[test]
fn site_temporal_policy_dynamic() {
    set_law(&[0, 2, 5], |items| {
        let policy = TemporalPolicy::dynamic(DynamicRuleId::from_raw(1), items.to_vec());
        Some(digest(&policy))
    });
}

/// Values are bit-exact and variant-tagged: two values share bytes only when they are equal
/// (bitwise for reals, so `-0.0` / `0.0` and NaN payloads differ).
#[test]
fn site_value_encoding() {
    let family = [
        Value::Float64(0.0),
        Value::Float64(-0.0),
        Value::Float64(1.0),
        Value::Float64(f64::NAN),
        Value::Float64(f64::from_bits(f64::NAN.to_bits() ^ 1)),
        Value::Int64(0),
        Value::Int64(1),
        Value::Bool(false),
        Value::Bool(true),
        Value::Category(0),
        Value::Category(1),
        Value::Label(Arc::from("")),
        Value::Label(Arc::from("1")),
        Value::symbolic_intervention(),
    ];
    injective(&family, Clone::clone, |value| Some(digest(value)));
}

/// Declared reads are sets: `PolicyReads::new` sorts and deduplicates both lists.
#[test]
fn site_policy_reads() {
    set_law(&trio(), |items| Some(digest(&PolicyReads::new(items.iter().copied(), []))));
    set_law(&[(v(5), 1_u32), (v(5), 2), (v(6), 1)], |items| {
        Some(digest(&PolicyReads::new([], items.iter().copied())))
    });
}
