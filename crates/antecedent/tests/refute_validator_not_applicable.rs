//! Per-run validator skips (`ValidationOutcome::NotApplicable`) must be visible to the
//! caller as diagnostics, not silently dropped by `ValidationSuite::reports_only`.
//!
//! See `crates/antecedent-validate/src/suite.rs` (`ValidationSuite::not_applicable_only`)
//! and `crates/antecedent/src/analysis/helpers.rs` (`validator_not_applicable_diagnostic`).
//! `manufacturing_temporal.rs` pins the temporal case (`OverlapRefuter` skipped on a
//! temporal-unfolded design under `RefuteSuite::Cheap`); this file pins a non-temporal
//! case.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]

use std::sync::Arc;

use antecedent::{RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, CausalSchemaBuilder, ConditionalEffectQuery, ExecutionContext,
    MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_graph::{Dag, DenseNodeId};

/// `ConditionalEffectQuery` executes through `ConditionalLinearAdjustment`, whose
/// estimator id is `"conditional.linear.adjustment"` — not `"linear.adjustment.ate"` —
/// so `ValidationSuite::run_one`'s `linear_ok` gate is false here and `Placebo` /
/// `RandomCommonCause` are each `NotApplicable` for this run (see
/// `crates/antecedent-validate/src/suite.rs`). Neither is a support-matrix refusal: the
/// same two validators run and pass on an unconditional ATE over the identical data
/// (`facade_gaps.rs::conditional_effect_via_causal_analysis` runs this estimator
/// successfully; only the refuters are gated).
fn conditional_query_fixture(n: usize) -> (TabularData, Dag, ConditionalEffectQuery) {
    let mut b = CausalSchemaBuilder::new();
    for name in ["t", "y", "w"] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let t: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
    let w: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
    let y: Vec<f64> =
        t.iter().zip(w.iter()).map(|(&ti, &wi)| 1.0 + 2.0 * ti + 0.5 * ti * wi).collect();
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(w), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let mut g = Dag::with_variables(3);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    let inner = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
        .with_effect_modifiers([VariableId::from_raw(2)]);
    (data, g, ConditionalEffectQuery::try_new(inner).unwrap())
}

#[test]
fn conditional_effect_placebo_and_rcc_skip_surfaces_as_diagnostics() {
    let (data, g, cq) = conditional_query_fixture(120);
    let analysis = Study::tabular(data)
        .graph(g)
        .query(CausalQuery::ConditionalEffect(cq))
        .refute(RefuteSuite::PlaceboAndRcc)
        .build()
        .unwrap();
    let result = analysis.run(&ExecutionContext::for_tests(1)).unwrap();
    assert!(result.estimate.ate.is_finite());

    // Both requested validators were NotApplicable this run, so the caller-visible
    // refutation list is empty...
    assert!(
        result.refutations.is_empty(),
        "expected no refutation reports (both validators NotApplicable); got {:?}",
        result.refutations.iter().map(|r| r.refuter.as_ref()).collect::<Vec<_>>()
    );

    // ...but the skip must not be invisible: one `refute.validator.not_applicable`
    // diagnostic per skipped validator, distinguishable from the support matrix's
    // permanent `not_applicable` refusal.
    let skips: Vec<_> = result
        .diagnostics
        .iter()
        .filter(|d| d.code.as_ref() == "refute.validator.not_applicable")
        .collect();
    assert_eq!(
        skips.len(),
        2,
        "expected one diagnostic each for Placebo and RandomCommonCause; got {:?}",
        result.diagnostics.iter().map(|d| d.code.as_ref()).collect::<Vec<_>>()
    );
    let validators: Vec<&str> = skips
        .iter()
        .flat_map(|d| d.fields.iter())
        .filter(|(k, _)| k.as_ref() == "validator")
        .map(|(_, v)| v.as_ref())
        .collect();
    assert!(validators.contains(&"placebo"), "expected a placebo skip diagnostic: {validators:?}");
    assert!(
        validators.contains(&"random_common_cause"),
        "expected a random_common_cause skip diagnostic: {validators:?}"
    );
    for d in &skips {
        assert!(
            d.message.contains("per-run") && d.message.contains("not a permanent"),
            "diagnostic message must distinguish this per-run skip from the support \
             matrix's permanent not_applicable state; got: {}",
            d.message
        );
    }
}
