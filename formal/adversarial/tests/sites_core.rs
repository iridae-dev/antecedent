//! Identity sites of antecedent-core (`formal/identity_sites.toml`, `core.*`) under generated
//! inputs: the category laws of `common` over the public constructors. The exhaustive
//! in-crate checks over every core site are the `unit:` checks the inventory names; these
//! cases add generated sizes and contents for the sites a public constructor reaches.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use antecedent_core::callback::CallbackOutput;
use antecedent_core::ids::DynamicRuleId;
use antecedent_core::intervention::TemporalPolicy;
use antecedent_core::value_expr::PolicyReads;
use antecedent_core::{
    AverageEffectQuery, CanonicalEncode, CanonicalWriter, PathSpecificEffectQuery, VariableId,
};
use proptest::prelude::*;

fn digest<T: CanonicalEncode>(value: &T) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.adversarial.core_site");
    value.encode(&mut writer);
    writer.finish()
}

fn variable() -> impl Strategy<Value = VariableId> {
    (5u32..40).prop_map(VariableId::from_raw)
}

proptest! {
    #![proptest_config(common::config())]

    /// Effect modifiers: reordering keeps the query identity; a repeat is kept apart (the
    /// estimators that read modifiers refuse a repeated one).
    #[test]
    fn site_core_average_effect_modifiers((items, shuffled) in common::permuted(variable(), 0..6)) {
        common::multiset_laws(&items, &shuffled, |list| {
            Some(digest(
                &AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                    .with_effect_modifiers(list.to_vec()),
            ))
        })?;
    }

    /// Path nodes: the same multiset law.
    #[test]
    fn site_core_path_nodes((items, shuffled) in common::permuted(variable(), 0..6)) {
        common::multiset_laws(&items, &shuffled, |list| {
            Some(digest(
                &PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(1))
                    .with_path_nodes(list.to_vec()),
            ))
        })?;
    }

    /// Declared policy reads are sets.
    #[test]
    fn site_core_policy_reads((items, shuffled) in common::distinct_permuted(variable(), 1..6)) {
        common::set_laws(&items, &shuffled, |list| {
            Some(digest(&PolicyReads::new(list.iter().copied(), [])))
        })?;
    }

    /// Dynamic schedules are sets of offsets.
    #[test]
    fn site_core_dynamic_schedule((items, shuffled) in common::distinct_permuted(-5i32..20, 1..6)) {
        common::set_laws(&items, &shuffled, |list| {
            Some(digest(&TemporalPolicy::dynamic(DynamicRuleId::from_raw(1), list.to_vec())))
        })?;
    }

    /// Callback levels are sets of finite reals.
    #[test]
    fn site_core_callback_levels(
        (items, shuffled) in common::distinct_permuted(-8i32..8, 1..6)
    ) {
        let reals = |list: &[i32]| list.iter().map(|&level| f64::from(level) / 2.0).collect::<Vec<f64>>();
        common::set_laws(&items, &shuffled, |list| {
            Some(digest(&CallbackOutput::levels(reals(list))))
        })?;
    }
}
