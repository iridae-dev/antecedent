//! Versioned identity for a lazy or chunked view of an owned source.
//!
//! Full content identity is version 1 of [`crate::storage::OwnedColumnarStorage::content_digest`]:
//! the digest of columns, masks, and weights, not a second hash of those bytes. A chunk or
//! lazy view uses its own domain and records the parent digest, a digest of the parent
//! contract (modality, validity policy, cohort order, unit, splits, row count, schema
//! names, and time index), the row range, and a version. Byte-identical storage under a
//! different column name, modality, cohort, split, or time index is therefore a different
//! parent and a different view. A projected column set has no full-source identity.
//!
//! This is not another owned snapshot. Callers keep [`OwnedSource`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::ops::Range;

use crate::error::DataError;
use crate::source::{
    CohortOrder, OwnedSource, RowChunk, SegmentedSource, SnapshotContract, SourceIdentity,
    SourceModality, ValidityPolicy,
};
use crate::temporal::SamplingRegularity;

/// Version of the full-content binding.
///
/// Version 1 is the bytes of [`crate::storage::OwnedColumnarStorage::content_digest`].
pub const FULL_CONTENT_VERSION: u16 = 1;

/// Version recorded on a chunk or lazy view.
pub const LAZY_VIEW_VERSION: u16 = 1;

/// BLAKE3 domain for a chunk or lazy view. Not the storage-digest domain.
const LAZY_DOMAIN: &str = "antecedent.data.lazy.v1";

/// BLAKE3 domain for the digest of a snapshot contract's scientific fields.
const CONTRACT_DOMAIN: &str = "antecedent.data.lazy.contract.v2";

/// BLAKE3 domain for the composite digest of a panel or multi-environment source.
const SEGMENT_DOMAIN: &str = "antecedent.data.lazy.segmented.v1";

/// BLAKE3 domain for the binding of a segmented source's parts.
const SEGMENT_BINDING_DOMAIN: &str = "antecedent.data.lazy.segmented.binding.v1";

/// Version 1 full-source digest.
///
/// The digest is the storage digest itself.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FullContentIdentity {
    version: u16,
    digest: [u8; 32],
    /// Digest of the contract the storage digest was recorded under.
    binding: [u8; 32],
}

/// Chunk or lazy view of a full source.
///
/// `digest` is domain-separated from [`FullContentIdentity`]. Equality with the parent
/// is not defined on this type; compare [`VersionedIdentity`] when both are in hand.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LazyViewIdentity {
    version: u16,
    parent: [u8; 32],
    binding: [u8; 32],
    start: usize,
    end: usize,
    digest: [u8; 32],
}

/// Full source, or a lazy view of one. The variants are not equal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VersionedIdentity {
    /// Columns, mask, and weights.
    Full(FullContentIdentity),
    /// Row range in the lazy domain.
    Lazy(LazyViewIdentity),
}

impl FullContentIdentity {
    /// Stamp version 1 onto a contract that already stores the storage digest.
    ///
    /// A projection is refused. This does not hash the columns again.
    ///
    /// # Errors
    ///
    /// The contract is [`SourceIdentity::Projected`].
    pub fn try_from_contract(contract: &SnapshotContract) -> Result<Self, DataError> {
        match contract.identity {
            SourceIdentity::Full(digest) => Ok(Self {
                version: FULL_CONTENT_VERSION,
                digest,
                binding: contract_binding(contract),
            }),
            SourceIdentity::Projected { .. } => Err(projection_refused()),
        }
    }

    /// Version 1 after checking the contract against `content_digest`.
    ///
    /// # Errors
    ///
    /// The source is a projection, or its recorded digest is not the storage digest.
    pub fn try_from_source(source: &OwnedSource) -> Result<Self, DataError> {
        let SourceIdentity::Full(recorded) = source.contract().identity else {
            return Err(projection_refused());
        };
        let scanned = source.borrow().storage().content_digest();
        if recorded != scanned {
            return Err(DataError::invalid_value(
                "snapshot digest",
                "must be version 1 of the storage content digest",
            ));
        }
        Ok(Self {
            version: FULL_CONTENT_VERSION,
            digest: scanned,
            binding: contract_binding(source.contract()),
        })
    }

    /// Format version. Version 1 is the storage digest.
    #[must_use]
    pub const fn version(&self) -> u16 {
        self.version
    }

    /// Storage digest bytes.
    #[must_use]
    pub const fn digest(&self) -> [u8; 32] {
        self.digest
    }

    /// Digest of the contract fields the storage digest does not cover.
    #[must_use]
    pub const fn binding(&self) -> [u8; 32] {
        self.binding
    }
}

impl LazyViewIdentity {
    /// Identity of `chunk` under the lazy domain.
    ///
    /// # Errors
    ///
    /// `full` is not version 1, or `chunk` names a different parent digest.
    pub fn from_chunk(full: &FullContentIdentity, chunk: &RowChunk) -> Result<Self, DataError> {
        if full.version != FULL_CONTENT_VERSION {
            return Err(DataError::invalid_value("full content identity", "must be version 1"));
        }
        if chunk.parent() != full.digest {
            return Err(DataError::mismatch("chunk parent and the full content digest"));
        }
        let rows = chunk.rows();
        Self::seal(full.digest, full.binding, rows.start, rows.end)
    }

    fn seal(
        parent: [u8; 32],
        binding: [u8; 32],
        start: usize,
        end: usize,
    ) -> Result<Self, DataError> {
        Ok(Self {
            version: LAZY_VIEW_VERSION,
            parent,
            binding,
            start,
            end,
            digest: lazy_digest(&parent, &binding, start, end)?,
        })
    }

    /// Whether `source` is still the source this view was taken of: the same content under
    /// the same contract, with the range still inside it. A setter that changes the weights,
    /// mask, a column, the schema, the time index or the event times makes this false.
    ///
    /// # Errors
    ///
    /// `source` is a projection or carries a digest that is not its storage digest.
    pub fn is_current_for(&self, source: &OwnedSource) -> Result<bool, DataError> {
        let full = FullContentIdentity::try_from_source(source)?;
        let Ok(chunk) = source.chunk(self.start, self.end) else {
            return Ok(false);
        };
        Ok(Self::from_chunk(&full, &chunk)? == *self)
    }

    /// Digest of the parent contract this view was taken under.
    #[must_use]
    pub const fn binding(&self) -> [u8; 32] {
        self.binding
    }

    /// Lazy-view version.
    #[must_use]
    pub const fn version(&self) -> u16 {
        self.version
    }

    /// Parent full-content digest.
    #[must_use]
    pub const fn parent(&self) -> [u8; 32] {
        self.parent
    }

    /// Half-open row range.
    #[must_use]
    pub const fn rows(&self) -> Range<usize> {
        self.start..self.end
    }

    /// Domain-separated digest. Not the parent storage digest.
    #[must_use]
    pub const fn digest(&self) -> [u8; 32] {
        self.digest
    }
}

impl FullContentIdentity {
    /// Version 1 composite over every part of a panel or multi-environment source.
    ///
    /// Each part is scanned (its recorded digest must be its storage digest). The
    /// composite digest covers the part order, each part's label, storage digest
    /// and contract binding (which carries its time index). It is not the digest of
    /// any one part, so it is the parent that chunks and increments bind to.
    ///
    /// # Errors
    ///
    /// A part is a projection or carries a digest that is not its storage digest.
    pub fn try_from_segmented(source: &SegmentedSource) -> Result<Self, DataError> {
        let mut digest = blake3::Hasher::new_derive_key(SEGMENT_DOMAIN);
        let mut binding = blake3::Hasher::new_derive_key(SEGMENT_BINDING_DOMAIN);
        let tag = modality_tag(source.modality());
        digest.update(&[tag]);
        binding.update(&[tag]);
        let count = source.parts().len() as u64;
        digest.update(&count.to_le_bytes());
        binding.update(&count.to_le_bytes());
        for part in source.parts() {
            let full = Self::try_from_source(part)?;
            let label = part.contract().unit.as_deref().unwrap_or("");
            for hasher in [&mut digest, &mut binding] {
                hasher.update(&(label.len() as u64).to_le_bytes());
                hasher.update(label.as_bytes());
                hasher.update(&full.binding);
            }
            digest.update(&full.digest);
        }
        Ok(Self {
            version: FULL_CONTENT_VERSION,
            digest: finish(&digest),
            binding: finish(&binding),
        })
    }
}

const fn modality_tag(modality: SourceModality) -> u8 {
    match modality {
        SourceModality::Tabular => 1,
        SourceModality::Panel => 2,
        SourceModality::MultiEnvironment => 3,
        SourceModality::TimeSeries => 4,
        SourceModality::EventSeries => 5,
    }
}

fn lazy_digest(
    parent: &[u8; 32],
    binding: &[u8; 32],
    start: usize,
    end: usize,
) -> Result<[u8; 32], DataError> {
    let start = u64::try_from(start).map_err(|_| DataError::overflow("chunk start as u64"))?;
    let end = u64::try_from(end).map_err(|_| DataError::overflow("chunk end as u64"))?;
    let mut hasher = blake3::Hasher::new_derive_key(LAZY_DOMAIN);
    hasher.update(&LAZY_VIEW_VERSION.to_le_bytes());
    hasher.update(parent);
    hasher.update(binding);
    hasher.update(&start.to_le_bytes());
    hasher.update(&end.to_le_bytes());
    Ok(finish(&hasher))
}

/// Digest of every scientific field of a contract except the content identity, which the
/// storage digest already carries.
fn contract_binding(contract: &SnapshotContract) -> [u8; 32] {
    fn text(hasher: &mut blake3::Hasher, value: &str) {
        hasher.update(&(value.len() as u64).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    fn word(hasher: &mut blake3::Hasher, value: usize) {
        hasher.update(&(value as u64).to_le_bytes());
    }
    let mut hasher = blake3::Hasher::new_derive_key(CONTRACT_DOMAIN);
    hasher.update(&[modality_tag(contract.modality)]);
    hasher.update(&[match contract.validity {
        ValidityPolicy::F64InvalidIffNan => 1,
        ValidityPolicy::NullDistinctFromNan => 2,
    }]);
    match &contract.cohort {
        CohortOrder::Storage => {
            hasher.update(&[1]);
        }
        CohortOrder::Labeled(label) => {
            hasher.update(&[2]);
            text(&mut hasher, label);
        }
    }
    match &contract.unit {
        None => {
            hasher.update(&[0]);
        }
        Some(unit) => {
            hasher.update(&[1]);
            text(&mut hasher, unit);
        }
    }
    word(&mut hasher, contract.splits.len());
    for split in &contract.splits {
        text(&mut hasher, &split.label);
        word(&mut hasher, split.rows.start);
        word(&mut hasher, split.rows.end);
    }
    word(&mut hasher, contract.rows);
    word(&mut hasher, contract.schema_names.len());
    for name in contract.schema_names.iter() {
        text(&mut hasher, name);
    }
    // The schema (value types, roles, units, measurement) and the event timestamps are not in
    // the storage digest; they are bound here.
    hasher.update(&contract.schema);
    match &contract.event_times {
        None => {
            hasher.update(&[0]);
        }
        Some(times) => {
            hasher.update(&[1]);
            hasher.update(times);
        }
    }
    match &contract.time_index {
        None => {
            hasher.update(&[0]);
        }
        Some(index) => {
            hasher.update(&[1]);
            match index.regularity {
                SamplingRegularity::Irregular => {
                    hasher.update(&[0]);
                }
                SamplingRegularity::Regular { interval_ns } => {
                    hasher.update(&[1]);
                    hasher.update(&interval_ns.to_le_bytes());
                }
            }
            word(&mut hasher, index.length);
        }
    }
    finish(&hasher)
}

fn finish(hasher: &blake3::Hasher) -> [u8; 32] {
    *hasher.finalize().as_bytes()
}

fn projection_refused() -> DataError {
    DataError::invalid_value("a projected column set", "is not the full-source identity")
}

#[cfg(test)]
mod tests {
    use crate::dataset::TabularData;

    use super::*;

    fn source(columns: &[(&str, &[f64])]) -> OwnedSource {
        let storage =
            TabularData::from_f64_columns(columns.iter().copied()).unwrap().storage().clone();
        OwnedSource::from_storage(storage, SourceModality::Tabular)
    }

    #[test]
    fn full_content_identity_is_storage_digest_version_1() {
        let owned = source(&[("x", &[1.0, 2.0, 3.0])]);
        let full = FullContentIdentity::try_from_source(&owned).unwrap();
        let stamped = FullContentIdentity::try_from_contract(owned.contract()).unwrap();
        assert_eq!(full.version(), FULL_CONTENT_VERSION);
        assert_eq!(full.digest(), owned.borrow().storage().content_digest());
        assert_eq!(stamped, full);

        let missing = source(&[("x", &[1.0, f64::NAN])]);
        assert_ne!(full.digest(), FullContentIdentity::try_from_source(&missing).unwrap().digest());
    }

    #[test]
    fn lazy_view_is_not_the_parent_full_identity() {
        let owned = source(&[("x", &[1.0, 2.0, 3.0, 4.0])]);
        let full = FullContentIdentity::try_from_source(&owned).unwrap();
        let tail = LazyViewIdentity::from_chunk(&full, &owned.chunk(2, 4).unwrap()).unwrap();
        let empty = LazyViewIdentity::from_chunk(&full, &owned.chunk(1, 1).unwrap()).unwrap();

        assert_eq!(tail.version(), LAZY_VIEW_VERSION);
        assert_eq!(tail.parent(), full.digest());
        assert_eq!(tail.rows(), 2..4);
        assert_eq!(empty.rows(), 1..1);
        assert_ne!(tail.digest(), full.digest());
        assert_ne!(VersionedIdentity::Full(full), VersionedIdentity::Lazy(tail));
        assert_ne!(empty, tail);

        let other = FullContentIdentity::try_from_source(&source(&[("x", &[9.0])])).unwrap();
        assert!(LazyViewIdentity::from_chunk(&other, &owned.chunk(0, 1).unwrap()).is_err());
    }

    #[test]
    fn same_bytes_under_a_different_contract_is_a_different_view() {
        let base = source(&[("x", &[1.0, 2.0, 3.0, 4.0])]);
        let renamed = source(&[("z", &[1.0, 2.0, 3.0, 4.0])]);
        let panel =
            OwnedSource::from_storage(base.borrow().storage().clone(), SourceModality::Panel);
        let cohort = base.clone().with_cohort(CohortOrder::Labeled("arrival".into()));
        let split = base
            .clone()
            .with_splits(vec![crate::source::SplitSpan { label: "fit".into(), rows: 0..2 }])
            .unwrap();
        let view = |owned: &OwnedSource| {
            let full = FullContentIdentity::try_from_source(owned).unwrap();
            LazyViewIdentity::from_chunk(&full, &owned.chunk(1, 3).unwrap()).unwrap()
        };
        let reference = view(&base);
        // Same storage digest, same row range: only the contract differs.
        assert_eq!(
            base.borrow().storage().content_digest(),
            panel.borrow().storage().content_digest()
        );
        assert_eq!(reference, view(&base.clone()));
        for other in [&renamed, &panel, &cohort, &split] {
            let other = view(other);
            assert_eq!(other.rows(), reference.rows());
            assert_ne!(other.digest(), reference.digest());
            assert_ne!(other.binding(), reference.binding());
        }
    }

    #[test]
    fn projected_columns_are_not_the_full_source_identity() {
        let owned = source(&[("x", &[1.0, 2.0]), ("y", &[3.0, 4.0])]);
        let full = FullContentIdentity::try_from_source(&owned).unwrap();
        let one = owned.project(&[0]).unwrap();
        let both = owned.project(&[0, 1]).unwrap();

        assert!(FullContentIdentity::try_from_contract(&one).is_err());
        assert!(FullContentIdentity::try_from_contract(&both).is_err());
        assert!(one.full_digest().is_none());
        assert!(both.full_digest().is_none());
        match both.identity {
            SourceIdentity::Projected { parent, .. } => assert_eq!(parent, full.digest()),
            SourceIdentity::Full(_) => panic!("projection reused the full identity"),
        }
    }

    #[test]
    fn time_index_changes_the_lazy_view() {
        use crate::temporal::{SamplingRegularity, TimeIndex};

        let owned = source(&[("x", &[1.0, 2.0, 3.0, 4.0])]);
        let timed = owned
            .clone()
            .with_time_index(TimeIndex {
                regularity: SamplingRegularity::Regular { interval_ns: 5 },
                length: 4,
            })
            .unwrap();
        let view = |source: &OwnedSource| {
            let full = FullContentIdentity::try_from_source(source).unwrap();
            let lazy = LazyViewIdentity::from_chunk(&full, &source.chunk(0, 2).unwrap()).unwrap();
            (full, lazy)
        };
        let (base_full, base_view) = view(&owned);
        let (timed_full, timed_view) = view(&timed);
        assert_eq!(base_full.digest(), timed_full.digest());
        assert_ne!(base_full.binding(), timed_full.binding());
        assert_ne!(base_view.digest(), timed_view.digest());
        assert_ne!(base_view.binding(), timed_view.binding());
    }

    fn series(values: &[f64], interval_ns: u64) -> crate::dataset::TimeSeriesData {
        crate::dataset::TimeSeriesData::from_f64_columns([("x", values)], interval_ns).unwrap()
    }

    fn panel(units: &[(u32, crate::dataset::TimeSeriesData)]) -> crate::panel::PanelData {
        let units: Vec<crate::panel::PanelUnit> = units
            .iter()
            .map(|(unit_id, series)| crate::panel::PanelUnit {
                unit_id: *unit_id,
                series: series.clone(),
            })
            .collect();
        crate::panel::PanelData::try_new(units).unwrap()
    }

    fn composite(source: &SegmentedSource) -> FullContentIdentity {
        FullContentIdentity::try_from_segmented(source).unwrap()
    }

    #[test]
    fn series_source_binds_the_series_time_index_like_the_setter() {
        use crate::temporal::{SamplingRegularity, TimeIndex};

        let data = series(&[1.0, 2.0, 3.0], 10);
        let from_series = OwnedSource::from_time_series(&data, SourceModality::TimeSeries).unwrap();
        let by_setter =
            OwnedSource::from_storage(data.storage().clone(), SourceModality::TimeSeries)
                .with_time_index(data.time_index().clone())
                .unwrap();
        assert_eq!(from_series.contract(), by_setter.contract());
        assert!(OwnedSource::from_time_series(&data, SourceModality::Tabular).is_err());

        let slower = series(&[1.0, 2.0, 3.0], 11);
        let other = OwnedSource::from_time_series(&slower, SourceModality::TimeSeries).unwrap();
        let (a, b) = (
            FullContentIdentity::try_from_source(&from_series).unwrap(),
            FullContentIdentity::try_from_source(&other).unwrap(),
        );
        assert_eq!(a.digest(), b.digest());
        assert_ne!(a.binding(), b.binding());
        let view = |owned: &OwnedSource, full: &FullContentIdentity| {
            LazyViewIdentity::from_chunk(full, &owned.chunk(0, 2).unwrap()).unwrap().digest()
        };
        assert_ne!(view(&from_series, &a), view(&other, &b));

        let irregular = crate::dataset::TimeSeriesData::try_new(
            data.storage().clone(),
            TimeIndex { regularity: SamplingRegularity::Irregular, length: 3 },
        )
        .unwrap();
        let irregular =
            OwnedSource::from_time_series(&irregular, SourceModality::TimeSeries).unwrap();
        assert_ne!(
            FullContentIdentity::try_from_source(&irregular).unwrap().binding(),
            a.binding()
        );
    }

    #[test]
    fn every_series_setter_invalidates_the_lazy_digest() {
        use std::sync::Arc;

        use crate::column::ValidityBitmap;

        let data = series(&[1.0, 2.0, 3.0, 4.0], 10);
        let observe = |data: &crate::dataset::TimeSeriesData| {
            let source = OwnedSource::from_time_series(data, SourceModality::TimeSeries).unwrap();
            let full = FullContentIdentity::try_from_source(&source).unwrap();
            let lazy = LazyViewIdentity::from_chunk(&full, &source.chunk(0, 3).unwrap()).unwrap();
            lazy.digest()
        };
        let base = observe(&data);
        assert_eq!(base, observe(&data.clone()));
        let weights = data.with_weights(Arc::from(vec![1.0, 2.0, 1.0, 2.0])).unwrap();
        let mask =
            data.with_analysis_mask(ValidityBitmap::from_bytes(vec![0b0111], 4).unwrap()).unwrap();
        let replaced = data
            .with_replaced_float(
                antecedent_core::VariableId::from_raw(0),
                Arc::from(vec![1.0, 2.0, 3.0, 5.0]),
            )
            .unwrap();
        let retimed = series(&[1.0, 2.0, 3.0, 4.0], 20);
        for changed in [&weights, &mask, &replaced, &retimed] {
            assert_ne!(observe(changed), base);
        }
    }

    #[test]
    fn source_setters_each_change_the_contract_binding() {
        use crate::temporal::{SamplingRegularity, TimeIndex};

        let base = source(&[("x", &[1.0, 2.0, 3.0, 4.0])]);
        let binding =
            |owned: &OwnedSource| FullContentIdentity::try_from_source(owned).unwrap().binding();
        let reference = binding(&base);
        let timed = base
            .clone()
            .with_time_index(TimeIndex {
                regularity: SamplingRegularity::Regular { interval_ns: 3 },
                length: 4,
            })
            .unwrap();
        let variants = [
            base.clone().with_unit("u0"),
            base.clone().with_cohort(CohortOrder::Labeled("arrival".into())),
            base.clone()
                .with_splits(vec![crate::source::SplitSpan { label: "fit".into(), rows: 0..2 }])
                .unwrap(),
            timed,
        ];
        let mut seen = vec![reference];
        for variant in &variants {
            let value = binding(variant);
            assert!(!seen.contains(&value));
            seen.push(value);
        }
        assert_eq!(
            binding(&base.clone().with_validity(ValidityPolicy::F64InvalidIffNan).unwrap()),
            reference
        );
    }

    #[test]
    fn panel_identity_follows_order_labels_content_and_each_time_index() {
        let (a, b) = (series(&[1.0, 2.0, 3.0], 10), series(&[4.0, 5.0, 6.0], 10));
        let base = SegmentedSource::from_panel(&panel(&[(0, a.clone()), (1, b.clone())]));
        let reference = composite(&base);
        assert_eq!(reference, composite(&base.clone()));
        assert_eq!(base.rows(), 6);
        assert_eq!(base.unit_id(1), Some(1));
        assert_eq!(base.unit_id(9), None);
        assert_eq!(base.modality(), SourceModality::Panel);
        // The composite is the digest of no single part.
        for part in base.parts() {
            assert_ne!(reference.digest(), part.borrow().storage().content_digest());
        }

        let reordered = SegmentedSource::from_panel(&panel(&[(1, b.clone()), (0, a.clone())]));
        let relabeled = SegmentedSource::from_panel(&panel(&[(0, a.clone()), (7, b.clone())]));
        let edited = SegmentedSource::from_panel(&panel(&[
            (0, a.clone()),
            (1, series(&[4.0, 5.0, 7.0], 10)),
        ]));
        let retimed = SegmentedSource::from_panel(&panel(&[
            (0, a.clone()),
            (1, series(&[4.0, 5.0, 6.0], 25)),
        ]));
        let fewer = SegmentedSource::from_panel(&panel(&[(0, a.clone())]));
        for other in [&reordered, &relabeled, &edited, &retimed, &fewer] {
            assert_ne!(composite(other).digest(), reference.digest());
        }
        // A time-index change leaves the storage digest of every part alone and
        // still moves the composite binding.
        let retimed_full = composite(&retimed);
        assert_ne!(retimed_full.binding(), reference.binding());
    }

    #[test]
    fn multi_environment_identity_differs_from_the_panel_of_the_same_series() {
        let (a, b) = (series(&[1.0, 2.0, 3.0], 10), series(&[4.0, 5.0, 6.0], 10));
        let environments =
            crate::multi_env::MultiEnvironmentData::try_new(vec![a.clone(), b.clone()]).unwrap();
        let env = SegmentedSource::from_multi_environment(&environments);
        let as_panel = SegmentedSource::from_panel(&panel(&[(0, a.clone()), (1, b.clone())]));
        assert_eq!(env.modality(), SourceModality::MultiEnvironment);
        assert_eq!(env.parts().len(), 2);
        assert_eq!(env.unit_id(0), None);
        assert_ne!(composite(&env).digest(), composite(&as_panel).digest());

        let reversed = crate::multi_env::MultiEnvironmentData::try_new(vec![b, a]).unwrap();
        assert_ne!(
            composite(&SegmentedSource::from_multi_environment(&reversed)).digest(),
            composite(&env).digest()
        );
        assert!(env.clone().with_part(5, env.parts()[0].clone()).is_err());
        let wrong = as_panel.parts()[0].clone();
        assert!(env.with_part(0, wrong).is_err());
    }
}
