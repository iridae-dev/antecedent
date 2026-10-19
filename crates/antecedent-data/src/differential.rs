//! Layout fixtures for the source snapshot.
//!
//! Owned tables, offset foreign buffers, and Arrow columns (feature `arrow`)
//! that carry the same scientific rows share selected rows and snapshot
//! identity. A different `NaN`, weight, order, or panel/environment boundary
//! has a different snapshot identity.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CategoryDomainId, CausalSchemaBuilder, MeasurementSpec, SmallRoleSet, ValueType, VariableId,
};

use crate::buffer::{F64Buffer, foreign_offset_f64};
use crate::categorical::{
    CategoricalColumn, CategoryCode, CategoryDomain, CategoryLevel, UnknownCategoryPolicy,
};
use crate::column::{ColumnView, Float64Column, OwnedColumn, ValidityBitmap};
use crate::dataset::{TabularData, TimeSeriesData};
use crate::missing::complete_rows;
use crate::multi_env::MultiEnvironmentData;
use crate::panel::{PanelData, PanelUnit};
use crate::source::{
    CohortOrder, OwnedSource, SourceIdentity, SourceModality, SplitSpan, ValidityPolicy,
};
use crate::storage::OwnedColumnarStorage;
use crate::table::TableView;

fn columns(pairs: &[(&str, &[f64])]) -> OwnedColumnarStorage {
    TabularData::from_f64_columns(pairs.iter().copied()).unwrap().storage().clone()
}

fn tabular(storage: OwnedColumnarStorage) -> OwnedSource {
    OwnedSource::from_storage(storage, SourceModality::Tabular)
}

fn selected(source: &OwnedSource) -> Vec<bool> {
    let borrowed = source.borrow();
    let storage = borrowed.storage();
    let ids: Vec<VariableId> =
        storage.schema().variables().iter().map(|variable| variable.id).collect();
    complete_rows(storage, storage.analysis_mask(), &ids).unwrap()
}

fn assert_same_science(left: &OwnedSource, right: &OwnedSource) {
    assert_eq!(left.contract(), right.contract());
    assert_eq!(selected(left), selected(right));
}

fn replace_float(
    n: usize,
    values: impl Into<F64Buffer>,
    validity: ValidityBitmap,
) -> OwnedColumnarStorage {
    let zeros = vec![0.0; n];
    let template = columns(&[("x", &zeros)]);
    let column = OwnedColumn::Float64(
        Float64Column::new(VariableId::from_raw(0), values, validity).unwrap(),
    );
    OwnedColumnarStorage::try_new(template.schema().clone(), vec![column], None, None).unwrap()
}

fn with_mask(storage: &OwnedColumnarStorage, mask: ValidityBitmap) -> OwnedColumnarStorage {
    OwnedColumnarStorage::try_new(
        storage.schema().clone(),
        storage.columns().to_vec(),
        Some(mask),
        storage.weights().map(Arc::from),
    )
    .unwrap()
}

fn with_weights(storage: &OwnedColumnarStorage, weights: &[f64]) -> OwnedColumnarStorage {
    OwnedColumnarStorage::try_new(
        storage.schema().clone(),
        storage.columns().to_vec(),
        storage.analysis_mask().cloned(),
        Some(Arc::from(weights)),
    )
    .unwrap()
}

/// All-valid bitmap of `n` bits with bit `row` cleared. Padding bits stay set.
fn mask_clearing(n: usize, row: usize) -> ValidityBitmap {
    let mut bytes = vec![0xFFu8; n.div_ceil(8)];
    bytes[row / 8] &= !(1 << (row % 8));
    ValidityBitmap::from_bytes(bytes, n).unwrap()
}

fn is_foreign(storage: &OwnedColumnarStorage) -> bool {
    match storage.column(VariableId::from_raw(0)).unwrap() {
        ColumnView::Float64(column) => column.values.is_foreign(),
        _ => panic!("expected a float64 column"),
    }
}

fn series(values: &[f64]) -> TimeSeriesData {
    TimeSeriesData::from_f64_columns([("x", values)], 1).unwrap()
}

fn named(series: &TimeSeriesData, modality: SourceModality, unit: &str) -> OwnedSource {
    OwnedSource::from_storage(series.storage().clone(), modality).with_unit(unit)
}

#[test]
fn owned_and_foreign_offset_layouts_share_selection_and_identity() {
    let logical = [1.0, f64::NAN, 3.0];
    let owned = tabular(columns(&[("x", &logical)]));
    // NumPy noncontiguous input is not imported. The foreign buffer is an offset
    // window into a longer allocation: the unit-stride slice that path accepts.
    let foreign = replace_float(
        logical.len(),
        foreign_offset_f64(vec![8.0, 1.0, f64::NAN, 3.0, 9.0], 1, logical.len()),
        ValidityBitmap::all_valid(logical.len()),
    );
    assert!(is_foreign(&foreign));
    assert_same_science(&owned, &tabular(foreign));
    assert_eq!(selected(&owned).as_slice(), &[true, false, true]);
}

#[test]
fn invalid_zero_matches_nan_and_a_finite_zero_does_not() {
    let nan = tabular(columns(&[("x", &[1.0, f64::NAN, 3.0])]));
    // A cleared validity bit whose payload is still 0 is `NaN` after construction.
    let invalid_zero = tabular(replace_float(3, vec![1.0, 0.0, 3.0], mask_clearing(3, 1)));
    assert_same_science(&nan, &invalid_zero);

    let finite_zero = tabular(columns(&[("x", &[1.0, 0.0, 3.0])]));
    assert_ne!(nan.contract(), finite_zero.contract());
    assert_eq!(selected(&finite_zero).as_slice(), &[true, true, true]);
    assert_ne!(selected(&nan), selected(&finite_zero));
}

#[test]
fn null_distinct_policy_is_not_the_nan_snapshot() {
    let nan = tabular(columns(&[("x", &[1.0, f64::NAN])]));
    let declared = nan.clone().with_validity(ValidityPolicy::NullDistinctFromNan);
    assert_eq!(nan.contract().validity, ValidityPolicy::F64InvalidIffNan);
    assert_ne!(nan.contract(), declared.contract());
    assert_eq!(nan.contract().full_digest(), declared.contract().full_digest());
    assert_eq!(selected(&nan), selected(&declared));
}

#[test]
fn analysis_mask_changes_selection_padding_does_not() {
    let storage = columns(&[("x", &[1.0, 2.0, 3.0])]);
    let unmasked = tabular(storage.clone());
    // Same three logical bits. The tight mask leaves padding clear; `mask_clearing` leaves it set.
    let tight = ValidityBitmap::from_bytes([0b0000_0101u8], 3).unwrap();
    let padded = mask_clearing(3, 1);
    assert_ne!(tight, padded);

    let tight_source = tabular(with_mask(&storage, tight));
    let padded_source = tabular(with_mask(&storage, padded));
    assert_same_science(&tight_source, &padded_source);
    assert_eq!(selected(&tight_source).as_slice(), &[true, false, true]);
    assert_ne!(unmasked.contract(), tight_source.contract());
    assert_ne!(selected(&unmasked), selected(&tight_source));

    let foreign = replace_float(
        3,
        foreign_offset_f64(vec![1.0, 2.0, 3.0, 4.0], 0, 3),
        ValidityBitmap::all_valid(3),
    );
    assert_same_science(&tight_source, &tabular(with_mask(&foreign, mask_clearing(3, 1))));
}

#[test]
fn projection_before_complete_case_changes_the_sample() {
    let x = [1.0, 2.0, 3.0];
    let y = [4.0, f64::NAN, 6.0];
    let storage = columns(&[("x", &x), ("y", &y)]);
    let full = tabular(storage.clone());
    let both = selected(&full);
    let only_x = complete_rows(&storage, None, &[VariableId::from_raw(0)]).unwrap();
    assert_eq!(both.as_slice(), &[true, false, true]);
    assert_eq!(only_x.as_slice(), &[true, true, true]);
    assert_ne!(both, only_x);

    let projected = full.project(&[0]).unwrap();
    assert!(projected.full_digest().is_none());
    let SourceIdentity::Projected { parent, .. } = projected.identity else {
        panic!("a one-column draw is a projection");
    };
    assert_eq!(parent, full.contract().full_digest().unwrap());
    assert_ne!(projected.identity, full.contract().identity);
}

#[test]
fn weights_change_identity_not_the_selected_rows() {
    let storage = columns(&[("x", &[1.0, 2.0, 3.0])]);
    let plain = tabular(storage.clone());
    let weighted = tabular(with_weights(&storage, &[1.0, 0.0, 2.0]));
    let again = tabular(with_weights(&storage, &[1.0, 0.0, 2.0]));
    assert_same_science(&weighted, &again);
    assert_ne!(plain.contract(), weighted.contract());
    assert_eq!(selected(&plain), selected(&weighted));
    assert_eq!(selected(&weighted).as_slice(), &[true, true, true]);

    let foreign = replace_float(
        3,
        foreign_offset_f64(vec![0.0, 1.0, 2.0, 3.0], 1, 3),
        ValidityBitmap::all_valid(3),
    );
    assert_same_science(&weighted, &tabular(with_weights(&foreign, &[1.0, 0.0, 2.0])));
}

#[test]
fn categorical_levels_change_identity_when_the_domain_differs() {
    fn categorical(labels: &[&str], ordered: bool, reference: Option<u32>) -> OwnedSource {
        let domain_id = CategoryDomainId::from_raw(7);
        let mut builder = CausalSchemaBuilder::new();
        builder
            .add_variable(
                "group",
                ValueType::Categorical,
                SmallRoleSet::empty(),
                None,
                Some(domain_id),
                MeasurementSpec::default(),
            )
            .unwrap();
        let schema = builder.build().unwrap();
        let levels: Arc<[CategoryLevel]> =
            labels.iter().map(|label| CategoryLevel { label: Arc::from(*label) }).collect();
        let domain = CategoryDomain::try_new(
            domain_id,
            levels,
            ordered,
            reference.map(CategoryCode::from_raw),
            UnknownCategoryPolicy::Fail,
        )
        .unwrap();
        let column = CategoricalColumn::try_new(
            VariableId::from_raw(0),
            Arc::from([
                CategoryCode::from_raw(0),
                CategoryCode::from_raw(1),
                CategoryCode::from_raw(0),
            ]),
            mask_clearing(3, 2),
            Arc::new(domain),
        )
        .unwrap();
        let storage = OwnedColumnarStorage::try_new(
            schema,
            vec![OwnedColumn::Categorical(column)],
            None,
            None,
        )
        .unwrap();
        tabular(storage)
    }

    let base = categorical(&["control", "treated"], false, Some(0));
    assert_same_science(&base, &categorical(&["control", "treated"], false, Some(0)));
    assert_eq!(selected(&base).as_slice(), &[true, true, false]);

    let swapped = categorical(&["treated", "control"], false, Some(0));
    let split_labels = categorical(&["ab", "c"], false, Some(0));
    let other_split = categorical(&["a", "bc"], false, Some(0));
    let unordered_reference = categorical(&["control", "treated"], false, None);
    let ordered = categorical(&["control", "treated"], true, Some(0));
    assert_ne!(base.contract(), swapped.contract());
    assert_ne!(split_labels.contract(), other_split.contract());
    assert_ne!(base.contract(), unordered_reference.contract());
    assert_ne!(base.contract(), ordered.contract());
    assert_eq!(selected(&base), selected(&swapped));
    assert_eq!(selected(&base), selected(&ordered));
}

#[test]
fn empty_and_tail_chunks_address_the_parent_source() {
    let values = [1.0, f64::NAN, 3.0, f64::NAN];
    let source = tabular(columns(&[("x", &values)]));
    let keep = selected(&source);
    assert_eq!(keep.as_slice(), &[true, false, true, false]);

    let tail = source.chunk(3, 4).unwrap();
    let empty_tail = source.chunk(4, 4).unwrap();
    let empty_head = source.chunk(0, 0).unwrap();
    assert_eq!(source.chunk(3, 4).unwrap(), tail);
    assert_eq!(tail.parent(), source.contract().full_digest().unwrap());
    assert_eq!(empty_tail.parent(), tail.parent());
    assert_eq!(empty_head.parent(), tail.parent());
    assert_ne!(empty_head, empty_tail);
    assert_eq!(&keep[tail.rows()], &[false]);
    assert!(keep[empty_tail.rows()].is_empty());
    assert!(keep[empty_head.rows()].is_empty());

    let tail_only = tabular(columns(&[("x", &[f64::NAN])]));
    assert_eq!(selected(&tail_only).as_slice(), &keep[tail.rows()]);
    assert_ne!(tail_only.contract().full_digest().unwrap(), tail.parent());

    let empty_table = tabular(columns(&[("x", &[])]));
    assert!(selected(&empty_table).is_empty());
    assert_ne!(empty_table.contract().full_digest().unwrap(), empty_tail.parent());
}

#[test]
fn row_order_changes_snapshot_identity() {
    let stored = tabular(columns(&[("x", &[1.0, 2.0, f64::NAN])]));
    let reversed = tabular(columns(&[("x", &[f64::NAN, 2.0, 1.0])]));
    assert_ne!(stored.contract(), reversed.contract());
    assert_eq!(selected(&stored).as_slice(), &[true, true, false]);
    assert_eq!(selected(&reversed).as_slice(), &[false, true, true]);

    let labeled = stored.clone().with_cohort(CohortOrder::Labeled(Arc::from("arrival")));
    assert_ne!(stored.contract(), labeled.contract());
    assert_eq!(stored.contract().full_digest(), labeled.contract().full_digest());
    assert_eq!(selected(&stored), selected(&labeled));
}

#[test]
fn unit_labels_change_snapshot_identity() {
    let stored = tabular(columns(&[("x", &[1.0, 2.0])]));
    let kilograms = stored.clone().with_unit("kg");
    let pounds = stored.with_unit("lb");
    assert_ne!(kilograms.contract(), pounds.contract());
    assert_eq!(kilograms.contract().full_digest(), pounds.contract().full_digest());
    assert_eq!(selected(&kilograms), selected(&pounds));
    assert_eq!(kilograms.contract(), kilograms.clone().with_unit("kg").contract());
}

#[test]
fn panel_and_environment_boundaries_change_identity() {
    let left = series(&[1.0, f64::NAN]);
    let right = series(&[3.0, 4.0]);
    let panel = PanelData::try_new(Arc::from([
        PanelUnit { unit_id: 0, series: left.clone() },
        PanelUnit { unit_id: 1, series: right.clone() },
    ]))
    .unwrap();
    let environments =
        MultiEnvironmentData::try_new(Arc::from([left.clone(), right.clone()])).unwrap();

    let panel_left = named(&panel.unit(0).unwrap().series, SourceModality::Panel, "u0");
    assert_same_science(&panel_left, &named(&left, SourceModality::Panel, "u0"));
    assert_eq!(selected(&panel_left).as_slice(), &[true, false]);

    let env_left =
        named(environments.environment(0).unwrap(), SourceModality::MultiEnvironment, "e0");
    assert_ne!(panel_left.contract(), env_left.contract());
    assert_eq!(panel_left.contract().full_digest(), env_left.contract().full_digest());
    assert_eq!(selected(&panel_left), selected(&env_left));

    let swapped = PanelData::try_new(Arc::from([
        PanelUnit { unit_id: 0, series: right.clone() },
        PanelUnit { unit_id: 1, series: left.clone() },
    ]))
    .unwrap();
    let swapped_left = named(&swapped.unit(0).unwrap().series, SourceModality::Panel, "u0");
    assert_ne!(panel_left.contract(), swapped_left.contract());
    assert_ne!(selected(&panel_left), selected(&swapped_left));

    let shifted = named(&series(&[1.0]), SourceModality::Panel, "u0");
    assert_ne!(panel_left.contract(), shifted.contract());
    assert_eq!(selected(&shifted).as_slice(), &[true]);

    let pooled = OwnedSource::from_storage(
        columns(&[("x", &[1.0, f64::NAN, 3.0, 4.0])]),
        SourceModality::Panel,
    );
    let at_two = pooled
        .clone()
        .with_splits(vec![
            SplitSpan { label: Arc::from("u0"), rows: 0..2 },
            SplitSpan { label: Arc::from("u1"), rows: 2..4 },
        ])
        .unwrap();
    let at_one = pooled
        .clone()
        .with_splits(vec![
            SplitSpan { label: Arc::from("u0"), rows: 0..1 },
            SplitSpan { label: Arc::from("u1"), rows: 1..4 },
        ])
        .unwrap();
    let as_environments = OwnedSource::from_storage(
        pooled.borrow().storage().clone(),
        SourceModality::MultiEnvironment,
    )
    .with_splits(at_two.contract().splits.clone())
    .unwrap();
    assert_ne!(at_two.contract(), at_one.contract());
    assert_ne!(at_two.contract(), as_environments.contract());
    assert_eq!(at_two.contract().full_digest(), at_one.contract().full_digest());
    assert_eq!(at_two.contract().full_digest(), as_environments.contract().full_digest());
    assert_eq!(selected(&at_two), selected(&at_one));

    let keep = selected(&pooled);
    assert_eq!(&keep[0..2], selected(&panel_left).as_slice());
    assert_eq!(&keep[2..4], selected(&named(&right, SourceModality::Panel, "u1")).as_slice());
    assert_ne!(&keep[0..1], selected(&panel_left).as_slice());

    let tail = at_two.chunk(2, 4).unwrap();
    let unit_right = named(&right, SourceModality::Panel, "u1");
    assert_eq!(tail.parent(), pooled.contract().full_digest().unwrap());
    assert_ne!(tail.parent(), unit_right.contract().full_digest().unwrap());
    assert_eq!(&keep[tail.rows()], selected(&unit_right).as_slice());

    let empty_env = named(
        MultiEnvironmentData::try_new(Arc::from([series(&[]), right]))
            .unwrap()
            .environment(0)
            .unwrap(),
        SourceModality::MultiEnvironment,
        "e0",
    );
    assert!(selected(&empty_env).is_empty());
    assert_ne!(empty_env.contract().full_digest(), env_left.contract().full_digest());
}

#[cfg(feature = "arrow")]
#[test]
fn arrow_null_and_offset_match_the_owned_table() {
    use arrow_array::{Float64Array, RecordBatch};
    use arrow_buffer::{NullBuffer, ScalarBuffer};
    use arrow_schema::{DataType, Field, Schema};

    use crate::arrow_adapter::{tabular_from_arrow_c_columns, tabular_from_record_batch};
    use crate::arrow_ffi::ArrowCColumn;

    let owned = tabular(columns(&[("x", &[1.0, f64::NAN, 3.0])]));

    let schema = Schema::new(vec![Field::new("x", DataType::Float64, true)]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(Float64Array::from(vec![Some(1.0), None, Some(3.0)]))],
    )
    .unwrap();
    let from_batch = tabular_from_record_batch(&batch).unwrap();
    assert!(!is_foreign(from_batch.data.storage()));
    assert_same_science(&owned, &tabular(from_batch.data.storage().clone()));

    let nan_null = Float64Array::new(
        ScalarBuffer::from(vec![1.0, f64::NAN, 3.0]),
        Some(NullBuffer::from(vec![true, false, true])),
    );
    let from_nan_null =
        tabular_from_arrow_c_columns(vec![ArrowCColumn::from_arrow_array("x", &nan_null).unwrap()])
            .unwrap();
    assert!(is_foreign(from_nan_null.data.storage()));
    assert_same_science(&owned, &tabular(from_nan_null.data.storage().clone()));

    // Exporters often leave 0 under a null. That placeholder is not an observation.
    let placeholder = Float64Array::from(vec![Some(1.0), None, Some(3.0)]);
    let from_placeholder = tabular_from_arrow_c_columns(vec![
        ArrowCColumn::from_arrow_array("x", &placeholder).unwrap(),
    ])
    .unwrap();
    assert!(!is_foreign(from_placeholder.data.storage()));
    assert_same_science(&owned, &tabular(from_placeholder.data.storage().clone()));

    // Slice of a longer Arrow column: the logical window, including a null whose
    // payload is not NaN. The importer copies that window into owned values.
    let longer = Float64Array::from(vec![Some(9.0), Some(1.0), None, Some(3.0), Some(8.0)]);
    let window = longer.slice(1, 3);
    let from_offset =
        tabular_from_arrow_c_columns(vec![ArrowCColumn::from_arrow_array("x", &window).unwrap()])
            .unwrap();
    assert!(!is_foreign(from_offset.data.storage()));
    assert_same_science(&owned, &tabular(from_offset.data.storage().clone()));
}
