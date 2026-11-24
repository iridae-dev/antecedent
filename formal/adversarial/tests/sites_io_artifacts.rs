//! Identity sites of `formal/identity_sites.toml` in antecedent-io-core artifact encoders,
//! checked through the public API with the category laws of `common`. Sites no public API
//! reaches are checked by in-crate tests (`unit:` checks in the inventory).
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use antecedent_io_core::canonical_json::canonical_json_digest;
use antecedent_io_core::cohort_decision::CovariateShiftDeclaration;
use antecedent_io_core::observation_query_artifact::{
    ObservationQueryWire, observation_query_identity,
};
use antecedent_io_core::temporal_recalc_artifact::{
    TEMPORAL_RECALC_FEATURE, TEMPORAL_RECALC_INFERENCE, TEMPORAL_RECALC_VERSION,
    TemporalFunctionalWire, TemporalRecalcArtifactWire, TemporalRecalcRequestWire,
};
use proptest::prelude::*;

fn recalc(edit: impl FnOnce(&mut TemporalRecalcRequestWire)) -> Option<String> {
    let mut wire = TemporalRecalcArtifactWire {
        version: TEMPORAL_RECALC_VERSION,
        required_features: vec![TEMPORAL_RECALC_FEATURE.into()],
        request: TemporalRecalcRequestWire {
            edges: vec![(0, 1), (1, 2)],
            bidirected: vec![],
            selection_targets: vec![0],
            horizon: 1,
            lag_alignment: vec![0, 1, 2],
            period: (0, 1),
            units: Vec::new(),
            snapshot_id: "snap".into(),
            initial_state: [0.5, 0.5],
            initial_state_id: "init".into(),
            functional: TemporalFunctionalWire::Response { sequence: vec![1] },
            benefit_per_unit: 1.0,
            cost: 0.0,
        },
        seed: 7,
        reports: Vec::new(),
        value: 0.0,
        inference: TEMPORAL_RECALC_INFERENCE.into(),
        premises_digest: String::new(),
        data_digest: String::new(),
    };
    edit(&mut wire.request);
    wire.expected_premises_digest().ok()
}

fn pair() -> impl Strategy<Value = (u32, u32)> {
    (0u32..6, 0u32..6).prop_filter("distinct endpoints", |(a, b)| a != b)
}

proptest! {
    #![proptest_config(common::config())]

    /// `io_core.temporal_recalc_artifact...expected_premises_digest`: bidirected pairs are
    /// unordered and, with the selection targets, sets (identity scheme v2).
    #[test]
    fn site_io_artifacts_temporal_recalc_premises(
        (pairs, shuffled) in common::distinct_permuted(pair(), 1..5),
        (targets, targets_shuffled) in common::distinct_permuted(0u32..8, 1..5),
        flips in prop::collection::vec(any::<bool>(), 5),
    ) {
        // Normalise the generated pairs so two of them are never one unordered pair.
        let mut unordered: Vec<(u32, u32)> = pairs.iter().map(|&(a, b)| (a.min(b), a.max(b))).collect();
        unordered.sort_unstable();
        unordered.dedup();
        prop_assume!(unordered.len() == pairs.len());
        common::set_laws(&pairs, &shuffled, |items| {
            recalc(|request| request.bidirected = items.to_vec())
        })?;
        let flipped: Vec<(u32, u32)> = pairs
            .iter()
            .zip(&flips)
            .map(|(&(a, b), &flip)| if flip { (b, a) } else { (a, b) })
            .collect();
        prop_assert_eq!(
            recalc(|request| request.bidirected = pairs.clone()),
            recalc(|request| request.bidirected = flipped)
        );
        common::set_laws(&targets, &targets_shuffled, |items| {
            recalc(|request| request.selection_targets = items.to_vec())
        })?;
    }

    /// `io_core.cohort_decision.CovariateShiftDeclaration.new`: the invariant modifiers are a
    /// set (canonical form v2).
    #[test]
    fn site_io_artifacts_covariate_shift_modifiers(
        (modifiers, shuffled) in common::distinct_permuted("[a-d]{1,3}", 1..5),
    ) {
        common::set_laws(&modifiers, &shuffled, |items| {
            let names: Vec<&str> = items.iter().map(String::as_str).collect();
            Some(CovariateShiftDeclaration::new("source", "target", &names))
        })?;
    }

    /// `io_core.observation_query_artifact.wire.ObservationQueryWire.canonical`: assignment
    /// lists are sets (canonical form v2).
    #[test]
    fn site_io_artifacts_observation_query_canonical(
        (observed, shuffled) in common::distinct_permuted(("[a-e]", 0u64..4), 1..5),
    ) {
        common::set_laws(&observed, &shuffled, |items| {
            let query = ObservationQueryWire::GaussianOutcome {
                outcome: "y".into(),
                interventions: vec![("a".into(), 1.0f64.to_bits())],
                observed: items.to_vec(),
            };
            observation_query_identity(None, &query.canonical(), "declared").ok()
        })?;
    }

    /// `io_core.canonical_json.write`: object entries are a set keyed by name, arrays are
    /// ordered, and text is framed (separator-bearing strings never merge).
    #[test]
    fn site_io_artifacts_canonical_json(
        (keys, shuffled) in common::distinct_permuted("[a-d]{1,2}", 1..5),
        items in prop::collection::vec(0i64..5, 2..5),
        i in any::<usize>(),
        j in any::<usize>(),
        left in prop::collection::vec(common::adversarial_text(), 0..3),
        right in prop::collection::vec(common::adversarial_text(), 0..3),
    ) {
        let object = |keys: &[String]| {
            let body: Vec<String> = keys.iter().map(|key| format!("{key:?}:{}", key.len())).collect();
            format!("{{{}}}", body.join(","))
        };
        common::set_laws(&keys, &shuffled, |keys| canonical_json_digest("site", &object(keys)).ok())?;
        common::ordered_laws(&items, i, j, |items| {
            let text: Vec<String> = items.iter().map(i64::to_string).collect();
            canonical_json_digest("site", &format!("[{}]", text.join(","))).ok()
        })?;
        // The alphabet has no quote or backslash, so each text is a JSON string as written.
        let array = |texts: &Vec<String>| {
            let quoted: Vec<String> = texts.iter().map(|text| format!("\"{text}\"")).collect();
            format!("[{}]", quoted.join(","))
        };
        common::injective(&left, &right, Clone::clone, |texts| {
            canonical_json_digest("site", &array(texts)).ok()
        })?;
    }
}
