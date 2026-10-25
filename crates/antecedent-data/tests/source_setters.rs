//! Every setter of every non-file modality yields a new unified source identity, and the
//! lazy views taken of the earlier source stop matching it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CausalSchema, CausalSchemaBuilder, VariableId};
use antecedent_data::column::ValidityBitmap;
use antecedent_data::lazy_identity::{FullContentIdentity, LazyViewIdentity};
use antecedent_data::source::{OwnedSource, SegmentedSource, SourceModality};
use antecedent_data::{
    EventData, MultiEnvironmentData, PanelData, PanelUnit, SamplingRegularity, TabularData,
    TimeIndex, TimeSeriesData,
};

const X: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
const Y: [f64; 4] = [4.0, 3.0, 2.0, 1.0];

fn weights(values: [f64; 4]) -> Arc<[f64]> {
    Arc::from(values.to_vec())
}

fn replacement() -> Arc<[f64]> {
    Arc::from(vec![9.0, 2.0, 3.0, 4.0])
}

fn drop_last_row() -> ValidityBitmap {
    ValidityBitmap::from_bytes(vec![0b0000_0111], 4).unwrap()
}

fn schema(treatment: bool) -> CausalSchema {
    let builder = CausalSchemaBuilder::new();
    let builder = if treatment {
        builder.continuous("x").treatment()
    } else {
        builder.continuous("x").finish()
    };
    builder.continuous("y").finish().build().unwrap()
}

fn table() -> TabularData {
    TabularData::from_f64_columns([("x", &X[..]), ("y", &Y[..])]).unwrap()
}

fn series(interval_ns: u64) -> TimeSeriesData {
    TimeSeriesData::from_f64_columns([("x", &X[..]), ("y", &Y[..])], interval_ns).unwrap()
}

fn index(interval_ns: u64) -> TimeIndex {
    TimeIndex { regularity: SamplingRegularity::Regular { interval_ns }, length: 4 }
}

fn events() -> EventData {
    EventData::try_new(table().storage().clone(), vec![0_i64, 10, 25, 40]).unwrap()
}

/// What a consumer holds after reading `source`.
struct Held {
    source: OwnedSource,
    view: LazyViewIdentity,
}

fn hold(source: &OwnedSource) -> Held {
    let full = FullContentIdentity::try_from_source(source).unwrap();
    let view = LazyViewIdentity::from_chunk(&full, &source.chunk(1, 3).unwrap()).unwrap();
    Held { source: source.clone(), view }
}

fn identity(source: &OwnedSource) -> ([u8; 32], [u8; 32]) {
    let full = FullContentIdentity::try_from_source(source).unwrap();
    (full.digest(), full.binding())
}

/// `after` is a new source: its identity differs, and what `before` holds no longer applies.
fn assert_invalidated(before: &Held, after: &OwnedSource, setter: &str) {
    assert_ne!(identity(&before.source), identity(after), "{setter}: identity unchanged");
    assert!(!before.view.is_current_for(after).unwrap(), "{setter}: stale lazy view accepted");
    // The earlier source is still itself.
    assert!(before.view.is_current_for(&before.source).unwrap());
    // A view of the new source is a different identity.
    let full = FullContentIdentity::try_from_source(after).unwrap();
    let fresh = LazyViewIdentity::from_chunk(&full, &after.chunk(1, 3).unwrap()).unwrap();
    assert_ne!(fresh.digest(), before.view.digest(), "{setter}: same lazy view");
}

#[test]
fn every_tabular_setter_yields_a_new_source_identity() {
    let data = table();
    let before = hold(&data.source());

    let w = weights([1.0, 2.0, 1.0, 2.0]);
    let weighted = data.source().with_weights(Arc::clone(&w)).unwrap();
    assert_invalidated(&before, &weighted, "with_weights");
    // The data-level setter and the source-level setter agree on the identity.
    assert_eq!(weighted.contract(), data.with_weights(w).unwrap().source().contract());

    let masked = data.source().with_analysis_mask(drop_last_row()).unwrap();
    assert_invalidated(&before, &masked, "with_analysis_mask");
    assert_eq!(
        masked.contract(),
        data.with_analysis_mask(drop_last_row()).unwrap().source().contract()
    );

    let id = VariableId::from_raw(0);
    let replaced = data.source().with_replaced_float(id, replacement()).unwrap();
    assert_invalidated(&before, &replaced, "with_replaced_float");
    assert_eq!(
        replaced.contract(),
        data.with_replaced_float(id, replacement()).unwrap().source().contract()
    );

    // The same bytes under another role are another source.
    let roled = data.source().with_schema(schema(true)).unwrap();
    let plain = data.source().with_schema(schema(false)).unwrap();
    assert_eq!(identity(&roled).0, identity(&plain).0, "roles are not in the storage digest");
    assert_ne!(roled.contract().schema, plain.contract().schema);
    assert_invalidated(&hold(&plain), &roled, "with_schema");
    assert_eq!(roled.contract(), data.with_schema(schema(true)).unwrap().source().contract());

    // Setting what is already there is the same source.
    let same = data
        .with_weights(weights([1.0, 2.0, 1.0, 2.0]))
        .unwrap()
        .source()
        .with_weights(weights([1.0, 2.0, 1.0, 2.0]))
        .unwrap();
    assert_eq!(same.contract(), weighted.contract());
}

#[test]
fn every_time_series_setter_yields_a_new_source_identity() {
    let data = series(10);
    let before = hold(&data.source(SourceModality::TimeSeries).unwrap());

    // A new cadence leaves the storage digest alone and changes the contract binding.
    let slower = data.with_time_index(index(11)).unwrap();
    let slower_source = slower.source(SourceModality::TimeSeries).unwrap();
    assert_eq!(identity(&before.source).0, identity(&slower_source).0);
    assert_invalidated(&before, &slower_source, "with_time_index");
    let source_level =
        data.source(SourceModality::TimeSeries).unwrap().with_time_index(index(11)).unwrap();
    assert_eq!(source_level.contract(), slower_source.contract());
    let irregular = TimeIndex { regularity: SamplingRegularity::Irregular, length: 4 };
    let irregular_source =
        data.with_time_index(irregular).unwrap().source(SourceModality::TimeSeries).unwrap();
    assert_invalidated(&before, &irregular_source, "with_time_index (regularity)");
    let short = TimeIndex { regularity: SamplingRegularity::Irregular, length: 3 };
    assert!(data.with_time_index(short).is_err());

    let weighted = data.with_weights(weights([1.0, 2.0, 1.0, 2.0])).unwrap();
    assert_invalidated(
        &before,
        &weighted.source(SourceModality::TimeSeries).unwrap(),
        "with_weights",
    );

    let masked = data.with_analysis_mask(drop_last_row()).unwrap();
    assert_invalidated(
        &before,
        &masked.source(SourceModality::TimeSeries).unwrap(),
        "with_analysis_mask",
    );

    let replaced = data.with_replaced_float(VariableId::from_raw(1), replacement()).unwrap();
    assert_invalidated(
        &before,
        &replaced.source(SourceModality::TimeSeries).unwrap(),
        "with_replaced_float",
    );

    let roled = data.with_schema(schema(true)).unwrap();
    assert_eq!(roled.time_index(), data.time_index());
    assert_invalidated(&before, &roled.source(SourceModality::TimeSeries).unwrap(), "with_schema");

    // A time series is not a tabular source.
    assert!(data.source(SourceModality::Tabular).is_err());
}

#[test]
fn event_timestamps_are_bound_into_the_event_source() {
    let data = events();
    let source = data.source();
    assert_eq!(source.contract().modality, SourceModality::EventSeries);
    assert!(source.contract().event_times.is_some());
    let before = hold(&source);

    // Same marks, other timestamps: the storage digest is unchanged, the source is not.
    let later = data.with_event_times(vec![0_i64, 10, 25, 41]).unwrap();
    let later_source = later.source();
    assert_eq!(identity(&source).0, identity(&later_source).0);
    assert_invalidated(&before, &later_source, "with_event_times");
    let source_level = source.clone().with_event_times(&[0, 10, 25, 41]).unwrap();
    assert_eq!(source_level.contract(), later_source.contract());

    // The same timestamps again are the same source.
    let same = data.with_event_times(vec![0_i64, 10, 25, 40]).unwrap().source();
    assert_eq!(same.contract(), source.contract());

    // Marks, weights and schema follow the content and the schema digests.
    let weighted = data
        .with_storage(
            table().with_weights(weights([1.0, 1.0, 2.0, 2.0])).unwrap().storage().clone(),
        )
        .unwrap();
    assert_invalidated(&before, &weighted.source(), "with_storage (weights)");
    let roled = data.with_storage(table().with_schema(schema(true)).unwrap().storage().clone());
    assert_invalidated(&before, &roled.unwrap().source(), "with_storage (schema)");

    // Timestamps belong to an event series only, and must be ordered and complete.
    assert!(table().source().with_event_times(&[0, 1, 2, 3]).is_err());
    assert!(source.clone().with_event_times(&[0, 10, 25]).is_err());
    assert!(source.with_event_times(&[0, 10, 5, 40]).is_err());
}

/// What a consumer holds after reading a segmented source.
struct HeldSegments {
    composite: FullContentIdentity,
}

fn hold_segments(source: &SegmentedSource) -> HeldSegments {
    HeldSegments { composite: FullContentIdentity::try_from_segmented(source).unwrap() }
}

fn assert_segments_invalidated(before: &HeldSegments, after: &SegmentedSource, setter: &str) {
    let now = FullContentIdentity::try_from_segmented(after).unwrap();
    assert_ne!(
        (before.composite.digest(), before.composite.binding()),
        (now.digest(), now.binding()),
        "{setter}: composite identity unchanged"
    );
}

fn panel() -> PanelData {
    PanelData::try_new(vec![
        PanelUnit { unit_id: 7, series: series(10) },
        PanelUnit { unit_id: 9, series: series(10) },
    ])
    .unwrap()
}

#[test]
fn every_panel_setter_yields_a_new_composite_identity() {
    let data = panel();
    let before = hold_segments(&data.source());

    let reweighted = series(10).with_weights(weights([1.0, 2.0, 1.0, 2.0])).unwrap();
    assert_segments_invalidated(
        &before,
        &data.with_unit_series(1, reweighted.clone()).unwrap().source(),
        "with_unit_series (weights)",
    );
    assert_segments_invalidated(
        &before,
        &data.with_unit_series(1, series(10).with_time_index(index(11)).unwrap()).unwrap().source(),
        "with_unit_series (time index)",
    );
    assert_segments_invalidated(
        &before,
        &data.with_unit_series(0, reweighted).unwrap().source(),
        "with_unit_series (other unit)",
    );
    assert_segments_invalidated(
        &before,
        &data.with_unit_id(1, 11).unwrap().source(),
        "with_unit_id",
    );
    // The part setter of the segmented source is the same invalidation.
    let part = data.source().parts()[1].clone().with_analysis_mask(drop_last_row()).unwrap();
    assert_segments_invalidated(
        &before,
        &data.source().with_part(1, part).unwrap(),
        "SegmentedSource::with_part",
    );
    assert!(data.with_unit_series(2, series(10)).is_err());
    assert!(data.with_unit_id(2, 1).is_err());

    // Re-setting what is there is the same composite.
    let same = data.with_unit_id(1, 9).unwrap().source();
    assert_eq!(
        FullContentIdentity::try_from_segmented(&same).unwrap(),
        before.composite,
        "an unchanged panel keeps its identity"
    );
}

#[test]
fn every_multi_environment_setter_yields_a_new_composite_identity() {
    let data = MultiEnvironmentData::try_new(vec![series(10), series(10)]).unwrap();
    let before = hold_segments(&data.source());

    assert_segments_invalidated(
        &before,
        &data
            .with_environment(1, series(10).with_analysis_mask(drop_last_row()).unwrap())
            .unwrap()
            .source(),
        "with_environment (mask)",
    );
    // Environments share one schema, so a role change on one environment alone is refused;
    // the change on every environment is a new composite.
    assert!(data.with_environment(0, series(10).with_schema(schema(true)).unwrap()).is_err());
    let roled = MultiEnvironmentData::try_new(vec![
        series(10).with_schema(schema(true)).unwrap(),
        series(10).with_schema(schema(true)).unwrap(),
    ])
    .unwrap();
    assert_segments_invalidated(&before, &roled.source(), "schema on every environment");
    assert!(data.with_environment(2, series(10)).is_err());
    let same = data.with_environment(1, series(10)).unwrap().source();
    assert_eq!(FullContentIdentity::try_from_segmented(&same).unwrap(), before.composite);
}
