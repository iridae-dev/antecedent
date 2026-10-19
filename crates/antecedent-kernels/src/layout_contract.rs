//! Layouts the portable kernels are measured on.
//!
//! A fused reduction is a unit-stride `f64` slice with no mask. A bitmap mask is a
//! packed bit vector, `1` = included, least-significant bit first, with one bit per
//! value. Column scaling is one contiguous `f64` column. These types refuse the
//! other layouts; they do not time the kernels.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::view::{BitMaskView, F64VectorView, ViewError};

/// Unit-stride `f64` values. [`F64VectorView::as_slice`] is `Some` for this layout.
#[derive(Clone, Copy, Debug)]
pub struct UnitStrideF64<'a> {
    values: &'a [f64],
}

impl<'a> UnitStrideF64<'a> {
    /// Borrow a contiguous column.
    #[must_use]
    pub const fn new(values: &'a [f64]) -> Self {
        Self { values }
    }

    /// The contiguous values.
    #[must_use]
    pub const fn as_slice(self) -> &'a [f64] {
        self.values
    }

    /// The view the fused reduction fast path accepts.
    #[must_use]
    pub const fn view(self) -> F64VectorView<'a> {
        F64VectorView::contiguous(self.values)
    }
}

/// Packed bitmap over a unit-stride column. Bit `i` lives at `bytes[i / 8] >> (i % 8)`.
#[derive(Clone, Copy, Debug)]
pub struct PackedBitmapMask<'a> {
    values: F64VectorView<'a>,
    mask: BitMaskView<'a>,
}

impl<'a> PackedBitmapMask<'a> {
    /// Pair `values` with a bitmap that has one bit per value.
    ///
    /// # Errors
    ///
    /// The byte buffer is shorter than `ceil(values.len() / 8)`.
    pub fn new(values: &'a [f64], bytes: &'a [u8]) -> Result<Self, ViewError> {
        let mask = BitMaskView::new(bytes, values.len())?;
        Ok(Self { values: F64VectorView::contiguous(values), mask })
    }

    /// Unit-stride values the mask covers.
    #[must_use]
    pub const fn values(self) -> F64VectorView<'a> {
        self.values
    }

    /// Included bits, least-significant bit first.
    #[must_use]
    pub const fn mask(self) -> BitMaskView<'a> {
        self.mask
    }
}

/// One contiguous `f64` column for in-place scaling.
#[derive(Debug)]
pub struct ContiguousColumn<'a> {
    values: &'a mut [f64],
}

impl<'a> ContiguousColumn<'a> {
    /// Borrow the column.
    #[must_use]
    pub const fn new(values: &'a mut [f64]) -> Self {
        Self { values }
    }

    /// The column, in order.
    #[must_use]
    pub const fn as_mut_slice(&mut self) -> &mut [f64] {
        self.values
    }
}

#[cfg(test)]
mod tests {
    use super::{ContiguousColumn, PackedBitmapMask, UnitStrideF64};

    #[test]
    fn unit_stride_view_is_a_slice() {
        let values = [1.0, 2.0, 3.0];
        let layout = UnitStrideF64::new(&values);
        assert_eq!(layout.view().as_slice(), Some(layout.as_slice()));
    }

    #[test]
    fn packed_bitmap_is_least_significant_bit_first() {
        let values = [0.0; 8];
        let layout = PackedBitmapMask::new(&values, &[0b0000_0001]).unwrap();
        assert!(layout.mask().get(0));
        assert!(!layout.mask().get(1));
        assert!(PackedBitmapMask::new(&values, &[]).is_err());
        let high = PackedBitmapMask::new(&values, &[0b1000_0000]).unwrap();
        assert!(high.mask().get(7));
        assert!(!high.mask().get(0));
    }

    #[test]
    fn contiguous_column_is_the_borrowed_slice() {
        let mut values = [1.0, 2.0];
        let mut column = ContiguousColumn::new(&mut values);
        assert_eq!(column.as_mut_slice(), [1.0, 2.0]);
    }
}
