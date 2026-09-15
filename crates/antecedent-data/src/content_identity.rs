//! Streaming identity of immutable storage contents; schema belongs to the
//! enclosing observation identity in antecedent-io.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{ColumnView, OwnedColumn, UnknownCategoryPolicy, ValidityBitmap};

struct Encoder(blake3::Hasher);

impl Encoder {
    fn byte(&mut self, value: u8) {
        self.0.update(&[value]);
    }

    fn len(&mut self, value: usize) {
        self.0.update(&u64::try_from(value).expect("length fits u64").to_le_bytes());
    }

    fn string(&mut self, value: &str) {
        self.len(value.len());
        self.0.update(value.as_bytes());
    }

    fn bitmap(&mut self, bitmap: &ValidityBitmap) {
        self.len(bitmap.len());
        // Hash logical bits, never padding bytes or the all-valid optimization.
        for start in (0..bitmap.len()).step_by(8) {
            let mut byte = 0;
            for bit in 0..8.min(bitmap.len() - start) {
                byte |= u8::from(bitmap.is_valid(start + bit)) << bit;
            }
            self.byte(byte);
        }
    }

    fn floats(&mut self, values: &[f64]) {
        self.len(values.len());
        for value in values {
            self.0.update(&value.to_bits().to_le_bytes());
        }
    }

    fn integers(&mut self, values: &[i64]) {
        self.len(values.len());
        for value in values {
            self.0.update(&value.to_le_bytes());
        }
    }

    fn column(&mut self, column: ColumnView<'_>) {
        self.0.update(&column.id().raw().to_le_bytes());
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
                self.0.update(&c.values);
            }
            ColumnView::Categorical(c) => {
                self.byte(3);
                self.len(c.codes.len());
                for code in c.codes.iter() {
                    self.0.update(&code.raw().to_le_bytes());
                }
                self.0.update(&c.domain.id.raw().to_le_bytes());
                self.len(c.domain.levels.len());
                for level in c.domain.levels.iter() {
                    self.string(&level.label);
                }
                self.byte(u8::from(c.domain.ordered));
                self.byte(u8::from(c.domain.reference.is_some()));
                if let Some(reference) = c.domain.reference {
                    self.0.update(&reference.raw().to_le_bytes());
                }
                match c.domain.unknown_policy {
                    UnknownCategoryPolicy::Fail => self.byte(0),
                    UnknownCategoryPolicy::MapToOther { other } => {
                        self.byte(1);
                        self.0.update(&other.raw().to_le_bytes());
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
}

pub(crate) fn storage_digest(
    columns: &[OwnedColumn],
    row_count: usize,
    mask: Option<&ValidityBitmap>,
    weights: Option<&[f64]>,
) -> [u8; 32] {
    let mut encoder = Encoder(blake3::Hasher::new_derive_key("antecedent.data.storage.v1"));
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
    *encoder.0.finalize().as_bytes()
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
}
