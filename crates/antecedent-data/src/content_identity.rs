//! Streaming identity of immutable storage contents; schema belongs to the
//! enclosing observation identity in antecedent-io.
//!
//! Values are staged into a fixed buffer and handed to BLAKE3 in bulk. The
//! hashed byte stream is the `antecedent.data.storage.v1` encoding; staging
//! only changes how many bytes each `update` call receives, never the bytes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{CategoryCode, ColumnView, OwnedColumn, UnknownCategoryPolicy, ValidityBitmap};

/// Bytes staged before one BLAKE3 `update` call.
const CHUNK: usize = 64 * 1024;

struct Encoder {
    hasher: blake3::Hasher,
    buf: Vec<u8>,
    /// Hasher `update` calls, observed by the bulk-hashing test.
    #[cfg(test)]
    updates: usize,
}

impl Encoder {
    fn new() -> Self {
        Self {
            hasher: blake3::Hasher::new_derive_key("antecedent.data.storage.v1"),
            buf: Vec::with_capacity(CHUNK),
            #[cfg(test)]
            updates: 0,
        }
    }

    fn update(&mut self, bytes: &[u8]) {
        #[cfg(test)]
        {
            self.updates += 1;
        }
        self.hasher.update(bytes);
    }

    fn flush(&mut self) {
        if !self.buf.is_empty() {
            let staged = std::mem::take(&mut self.buf);
            self.update(&staged);
            self.buf = staged;
            self.buf.clear();
        }
    }

    fn put(&mut self, bytes: &[u8]) {
        if self.buf.len() + bytes.len() > CHUNK {
            self.flush();
        }
        if bytes.len() >= CHUNK {
            self.update(bytes);
        } else {
            self.buf.extend_from_slice(bytes);
        }
    }

    /// Stage fixed-width little-endian words without one hasher call per value.
    fn words<const W: usize, T: Copy>(&mut self, values: &[T], to_le: impl Fn(T) -> [u8; W]) {
        for chunk in values.chunks(CHUNK / W) {
            if self.buf.len() + chunk.len() * W > CHUNK {
                self.flush();
            }
            for &value in chunk {
                self.buf.extend_from_slice(&to_le(value));
            }
        }
    }

    fn byte(&mut self, value: u8) {
        self.put(&[value]);
    }

    fn len(&mut self, value: usize) {
        self.put(&u64::try_from(value).expect("length fits u64").to_le_bytes());
    }

    fn string(&mut self, value: &str) {
        self.len(value.len());
        self.put(value.as_bytes());
    }

    fn bitmap(&mut self, bitmap: &ValidityBitmap) {
        self.len(bitmap.len());
        // Logical bits, least-significant first: whole stored bytes, then the
        // tail byte with padding bits cleared. The all-valid optimization and
        // padding never reach the digest.
        let full = bitmap.len() / 8;
        let tail = bitmap.len() % 8;
        let bytes = bitmap.raw_bytes();
        self.put(&bytes[..full]);
        if tail > 0 {
            self.byte(bytes[full] & ((1u8 << tail) - 1));
        }
    }

    fn floats(&mut self, values: &[f64]) {
        self.len(values.len());
        self.words(values, |value: f64| value.to_bits().to_le_bytes());
    }

    fn integers(&mut self, values: &[i64]) {
        self.len(values.len());
        self.words(values, i64::to_le_bytes);
    }

    fn column(&mut self, column: ColumnView<'_>) {
        self.put(&column.id().raw().to_le_bytes());
        self.bitmap(column.validity());
        match column {
            ColumnView::Float64(c) => {
                self.byte(0);
                self.floats(c.values.as_slice());
            }
            ColumnView::Int64(c) => {
                self.byte(1);
                self.integers(&c.values);
            }
            ColumnView::Boolean(c) => {
                self.byte(2);
                self.len(c.values.len());
                self.put(&c.values);
            }
            ColumnView::Categorical(c) => {
                self.byte(3);
                self.len(c.codes.len());
                self.words(&c.codes, |code: CategoryCode| code.raw().to_le_bytes());
                self.put(&c.domain.id.raw().to_le_bytes());
                self.len(c.domain.levels.len());
                for level in c.domain.levels.iter() {
                    self.string(&level.label);
                }
                self.byte(u8::from(c.domain.ordered));
                self.byte(u8::from(c.domain.reference.is_some()));
                if let Some(reference) = c.domain.reference {
                    self.put(&reference.raw().to_le_bytes());
                }
                match c.domain.unknown_policy {
                    UnknownCategoryPolicy::Fail => self.byte(0),
                    UnknownCategoryPolicy::MapToOther { other } => {
                        self.byte(1);
                        self.put(&other.raw().to_le_bytes());
                    }
                }
            }
            ColumnView::Timestamp(c) => {
                self.byte(4);
                self.integers(&c.values_ns);
            }
            ColumnView::FixedVector(c) => {
                self.byte(5);
                self.len(c.dim);
                self.floats(&c.values);
            }
        }
    }

    fn finish(mut self) -> [u8; 32] {
        self.flush();
        *self.hasher.finalize().as_bytes()
    }
}

pub(crate) fn storage_digest(
    columns: &[OwnedColumn],
    row_count: usize,
    mask: Option<&ValidityBitmap>,
    weights: Option<&[f64]>,
) -> [u8; 32] {
    encode_storage(columns, row_count, mask, weights).finish()
}

fn encode_storage(
    columns: &[OwnedColumn],
    row_count: usize,
    mask: Option<&ValidityBitmap>,
    weights: Option<&[f64]>,
) -> Encoder {
    let mut encoder = Encoder::new();
    encoder.len(row_count);
    encoder.len(columns.len());
    for column in columns {
        encoder.column(column.as_view());
    }
    encoder.byte(u8::from(mask.is_some()));
    if let Some(mask) = mask {
        encoder.bitmap(mask);
    }
    encoder.byte(u8::from(weights.is_some()));
    if let Some(weights) = weights {
        encoder.floats(weights);
    }
    encoder
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BooleanColumn, CategoricalColumn, CategoryCode, CategoryDomain, CategoryLevel,
        FixedVectorColumn, Float64Column, Int64Column, TimestampColumn,
    };
    use antecedent_core::{CategoryDomainId, VariableId};
    use std::sync::Arc;

    fn hash(column: OwnedColumn) -> [u8; 32] {
        storage_digest(&[column], 2, None, None)
    }

    #[test]
    fn contents_order_masks_weights_and_validity_are_distinct() {
        let data = crate::testing::float_series(9, 2);
        let base = data.storage();
        assert_eq!(base.content_digest(), base.clone().content_digest());
        let rebuilt = crate::OwnedColumnarStorage::try_new(
            base.schema().clone(),
            base.columns().to_vec(),
            None,
            None,
        )
        .unwrap();
        assert_eq!(base.content_digest(), rebuilt.content_digest());
        let changed = data
            .with_replaced_float(
                VariableId::from_raw(0),
                Arc::from([8., 7., 6., 5., 4., 3., 2., 1., 0.]),
            )
            .unwrap();
        assert_ne!(base.content_digest(), changed.storage().content_digest());
        assert_ne!(
            base.content_digest(),
            crate::testing::float_series_with_gap(9, 2, 3).storage().content_digest()
        );
        assert_ne!(
            base.content_digest(),
            crate::testing::float_series_with_mask(9, 2, 3).storage().content_digest()
        );
        let weighted = crate::OwnedColumnarStorage::try_new(
            base.schema().clone(),
            base.columns().to_vec(),
            None,
            Some(Arc::from([2.; 9])),
        )
        .unwrap();
        assert_ne!(base.content_digest(), weighted.content_digest());
        let mask = ValidityBitmap::from_bytes([0xff, 0x01].as_slice(), 9).unwrap();
        let padded = ValidityBitmap::from_bytes([0xff, 0xff, 0x77].as_slice(), 9).unwrap();
        assert_eq!(
            storage_digest(base.columns(), 9, Some(&mask), None),
            storage_digest(base.columns(), 9, Some(&padded), None)
        );
    }

    #[test]
    fn typed_values_keep_exact_bits_and_category_domains() {
        let id = VariableId::from_raw(0);
        let valid = ValidityBitmap::all_valid(2);
        let float = |v: [f64; 2]| {
            OwnedColumn::Float64(Float64Column::new(id, Arc::from(v), valid.clone()).unwrap())
        };
        assert_ne!(hash(float([0., 1.])), hash(float([-0., 1.])));
        let integer = |v: [i64; 2]| {
            OwnedColumn::Int64(Int64Column::new(id, Arc::from(v), valid.clone()).unwrap())
        };
        // These adjacent integers would collide if coerced through f64.
        assert_ne!(
            hash(integer([9_007_199_254_740_992, 1])),
            hash(integer([9_007_199_254_740_993, 1]))
        );
        assert_ne!(hash(integer([0, 1])), hash(float([0., 1.])));
        assert_ne!(
            hash(integer([0, 1])),
            hash(OwnedColumn::Timestamp(
                TimestampColumn::new(id, Arc::from([0, 1]), valid.clone()).unwrap()
            ))
        );
        assert_ne!(
            hash(OwnedColumn::Boolean(
                BooleanColumn::new(id, Arc::<[u8]>::from([0, 1]), valid.clone()).unwrap()
            )),
            hash(integer([0, 1]))
        );
        assert_ne!(
            hash(OwnedColumn::FixedVector(
                FixedVectorColumn::new(id, 1, Arc::from([0., 1.]), valid.clone()).unwrap()
            )),
            hash(float([0., 1.]))
        );
        let categorical = |labels: [&str; 2]| {
            let domain = CategoryDomain::try_new(
                CategoryDomainId::from_raw(0),
                Arc::from(labels.map(|label| CategoryLevel { label: Arc::from(label) })),
                false,
                None,
                UnknownCategoryPolicy::Fail,
            )
            .unwrap();
            OwnedColumn::Categorical(
                CategoricalColumn::try_new(
                    id,
                    Arc::from([CategoryCode::from_raw(0), CategoryCode::from_raw(1)]),
                    valid.clone(),
                    Arc::new(domain),
                )
                .unwrap(),
            )
        };
        assert_ne!(hash(categorical(["a", "bc"])), hash(categorical(["ab", "c"])));
        assert_ne!(hash(categorical(["a", "b"])), hash(categorical(["b", "a"])));
    }

    use crate::TableView;

    /// The per-value `antecedent.data.storage.v1` encoding, one hasher call per
    /// field. Bulk staging must reproduce these bytes exactly.
    fn reference_digest(
        columns: &[OwnedColumn],
        row_count: usize,
        mask: Option<&ValidityBitmap>,
        weights: Option<&[f64]>,
    ) -> [u8; 32] {
        fn len(h: &mut blake3::Hasher, value: usize) {
            h.update(&(value as u64).to_le_bytes());
        }
        fn bitmap(h: &mut blake3::Hasher, bitmap: &ValidityBitmap) {
            len(h, bitmap.len());
            for start in (0..bitmap.len()).step_by(8) {
                let mut byte = 0;
                for bit in 0..8.min(bitmap.len() - start) {
                    byte |= u8::from(bitmap.is_valid(start + bit)) << bit;
                }
                h.update(&[byte]);
            }
        }
        fn floats(h: &mut blake3::Hasher, values: &[f64]) {
            len(h, values.len());
            for value in values {
                h.update(&value.to_bits().to_le_bytes());
            }
        }
        fn integers(h: &mut blake3::Hasher, values: &[i64]) {
            len(h, values.len());
            for value in values {
                h.update(&value.to_le_bytes());
            }
        }
        let mut h = blake3::Hasher::new_derive_key("antecedent.data.storage.v1");
        len(&mut h, row_count);
        len(&mut h, columns.len());
        for column in columns {
            let column = column.as_view();
            h.update(&column.id().raw().to_le_bytes());
            bitmap(&mut h, column.validity());
            match column {
                ColumnView::Float64(c) => {
                    h.update(&[0]);
                    floats(&mut h, c.values.as_slice());
                }
                ColumnView::Int64(c) => {
                    h.update(&[1]);
                    integers(&mut h, &c.values);
                }
                ColumnView::Boolean(c) => {
                    h.update(&[2]);
                    len(&mut h, c.values.len());
                    h.update(&c.values);
                }
                ColumnView::Categorical(c) => {
                    h.update(&[3]);
                    len(&mut h, c.codes.len());
                    for code in c.codes.iter() {
                        h.update(&code.raw().to_le_bytes());
                    }
                    h.update(&c.domain.id.raw().to_le_bytes());
                    len(&mut h, c.domain.levels.len());
                    for level in c.domain.levels.iter() {
                        len(&mut h, level.label.len());
                        h.update(level.label.as_bytes());
                    }
                    h.update(&[u8::from(c.domain.ordered)]);
                    h.update(&[u8::from(c.domain.reference.is_some())]);
                    if let Some(reference) = c.domain.reference {
                        h.update(&reference.raw().to_le_bytes());
                    }
                    match c.domain.unknown_policy {
                        UnknownCategoryPolicy::Fail => {
                            h.update(&[0]);
                        }
                        UnknownCategoryPolicy::MapToOther { other } => {
                            h.update(&[1]);
                            h.update(&other.raw().to_le_bytes());
                        }
                    }
                }
                ColumnView::Timestamp(c) => {
                    h.update(&[4]);
                    integers(&mut h, &c.values_ns);
                }
                ColumnView::FixedVector(c) => {
                    h.update(&[5]);
                    len(&mut h, c.dim);
                    floats(&mut h, &c.values);
                }
            }
        }
        h.update(&[u8::from(mask.is_some())]);
        if let Some(mask) = mask {
            bitmap(&mut h, mask);
        }
        h.update(&[u8::from(weights.is_some())]);
        if let Some(weights) = weights {
            floats(&mut h, weights);
        }
        *h.finalize().as_bytes()
    }

    /// Mixed-type columns long enough to cross several staging chunks, with a
    /// partial tail byte whose padding bits are set.
    fn mixed_columns(n: usize) -> (Vec<OwnedColumn>, ValidityBitmap, Vec<f64>) {
        let mut raw = vec![0b1011_0110u8; n.div_ceil(8)];
        if let Some(last) = raw.last_mut() {
            *last = 0xff;
        }
        let validity = ValidityBitmap::from_bytes(raw, n).unwrap();
        let index = |i: usize| u32::try_from(i).unwrap();
        let floats: Vec<f64> = (0..n).map(|i| f64::from(index(i)).sin() * 1e3).collect();
        let ints: Vec<i64> = (0..n).map(|i| i64::from(index(i)) * 7_919 - 1_000_000_007).collect();
        let bools: Vec<u8> = (0..n).map(|i| u8::from(i % 3 == 0)).collect();
        let stamps: Vec<i64> =
            (0..n).map(|i| 1_700_000_000_000_000_000 + i64::from(index(i))).collect();
        let vectors: Vec<f64> = (0..2 * n).map(|i| f64::from(index(i)) / 3.0).collect();
        let domain = CategoryDomain::try_new(
            CategoryDomainId::from_raw(4),
            Arc::from([
                CategoryLevel { label: Arc::from("low") },
                CategoryLevel { label: Arc::from("high") },
            ]),
            true,
            Some(CategoryCode::from_raw(0)),
            UnknownCategoryPolicy::MapToOther { other: CategoryCode::from_raw(1) },
        )
        .unwrap();
        let codes: Vec<CategoryCode> =
            (0..n).map(|i| CategoryCode::from_raw(u32::from(i % 2 == 1))).collect();
        let columns = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(floats), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Int64(
                Int64Column::new(VariableId::from_raw(1), Arc::from(ints), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Boolean(
                BooleanColumn::new(
                    VariableId::from_raw(2),
                    Arc::<[u8]>::from(bools),
                    validity.clone(),
                )
                .unwrap(),
            ),
            OwnedColumn::Timestamp(
                TimestampColumn::new(VariableId::from_raw(3), Arc::from(stamps), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::FixedVector(
                FixedVectorColumn::new(
                    VariableId::from_raw(4),
                    2,
                    Arc::from(vectors),
                    validity.clone(),
                )
                .unwrap(),
            ),
            OwnedColumn::Categorical(
                CategoricalColumn::try_new(
                    VariableId::from_raw(5),
                    Arc::from(codes),
                    validity.clone(),
                    Arc::new(domain),
                )
                .unwrap(),
            ),
        ];
        let weights: Vec<f64> = (0..n).map(|i| 1.0 + f64::from(index(i) % 5)).collect();
        (columns, validity, weights)
    }

    #[test]
    fn bulk_staging_reproduces_the_per_value_encoding() {
        for n in [0, 1, 7, 9, 8_193, 70_001] {
            let (columns, mask, weights) = mixed_columns(n);
            assert_eq!(
                storage_digest(&columns, n, Some(&mask), Some(&weights)),
                reference_digest(&columns, n, Some(&mask), Some(&weights)),
                "n={n}"
            );
            assert_eq!(
                storage_digest(&columns, n, None, None),
                reference_digest(&columns, n, None, None),
                "n={n}"
            );
        }
    }

    #[test]
    fn large_columns_hash_in_bulk_not_per_value() {
        let n = 200_000;
        let (columns, _, weights) = mixed_columns(n);
        let encoder = encode_storage(&columns, n, None, Some(&weights));
        // ~8·n bytes per float column; one call per value would be > n calls.
        let bytes = 8 * n * 5 + 4 * n + n + 2 * n;
        assert!(
            encoder.updates <= bytes / CHUNK + 64,
            "{} hasher updates for {bytes} bytes",
            encoder.updates
        );
    }
}
