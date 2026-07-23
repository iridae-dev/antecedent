//! Growable aligned f64 scratch buffers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Aligned growable `f64` buffer for design-matrix materialization.
#[derive(Clone, Debug, Default)]
pub struct AlignedBuffer<T> {
    data: Vec<T>,
}

impl<T: Copy + Default> AlignedBuffer<T> {
    /// Empty buffer.
    #[must_use]
    pub const fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Ensure at least `len` elements (grows, never shrinks; fills with default).
    pub fn resize(&mut self, len: usize) {
        if self.data.len() < len {
            self.data.resize(len, T::default());
        }
    }

    /// Borrow as slice of logical length `len` (must have been resized).
    ///
    /// # Panics
    ///
    /// When `len > self.data.len()`.
    #[must_use]
    pub fn as_slice(&self, len: usize) -> &[T] {
        &self.data[..len]
    }

    /// Mutable borrow of logical length `len`.
    ///
    /// # Panics
    ///
    /// When `len > self.data.len()`.
    pub fn as_mut_slice(&mut self, len: usize) -> &mut [T] {
        &mut self.data[..len]
    }

    /// Underlying capacity in elements.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Current allocated length (may exceed logical use).
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl AlignedBuffer<f64> {
    /// Ensure capacity then return mutable slice of length `len`.
    pub fn prepare_mut(&mut self, len: usize) -> &mut [f64] {
        self.resize(len);
        self.as_mut_slice(len)
    }
}
