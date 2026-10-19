//! Contiguous f64 column buffers — owned or foreign-backed.
//!
//! Foreign buffers never expose Arrow types; the owner is type-erased.
//! Unsafe code here is limited to pointer→slice projection under an owner
//! lifetime contract.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    unsafe_code,
    reason = "pointer-to-slice projection under the owner lifetime contract; each unsafe block states its safety argument"
)]

use std::ops::Deref;
use std::sync::Arc;

/// Contiguous `f64` values for a column.
#[derive(Clone, Debug)]
pub struct F64Buffer {
    inner: F64BufferInner,
    /// Known to hold no `NaN` (set by a scan that found none; the values are
    /// immutable, so the fact survives clones and re-wraps).
    nan_free: bool,
}

#[derive(Clone, Debug)]
enum F64BufferInner {
    Owned(Arc<[f64]>),
    Foreign(ForeignF64Buffer),
}

/// Foreign-backed f64 slice kept alive by an opaque owner.
#[derive(Clone, Debug)]
pub struct ForeignF64Buffer {
    owner: Arc<dyn ForeignBufferOwner>,
    ptr: *const f64,
    len: usize,
}

/// Type-erased owner that keeps foreign memory valid.
///
/// Implement this for the type that owns the allocation. A blanket impl would
/// accept any `Debug` value, including one that does not keep the bytes alive.
pub trait ForeignBufferOwner: Send + Sync + std::fmt::Debug {}

// SAFETY: `ptr` is valid for `len` elements while `owner` is alive; owner is Send+Sync.
unsafe impl Send for ForeignF64Buffer {}
unsafe impl Sync for ForeignF64Buffer {}

impl ForeignF64Buffer {
    /// Construct from a validated pointer and length.
    ///
    /// # Safety
    ///
    /// - `ptr` must be valid for reads of `len` `f64` values for the lifetime of `owner`.
    /// - The pointed-to memory must be properly aligned for `f64`.
    /// - No mutable aliases may exist while this buffer (or clones) is live.
    #[must_use]
    pub unsafe fn new(owner: Arc<dyn ForeignBufferOwner>, ptr: *const f64, len: usize) -> Self {
        Self { owner, ptr, len }
    }

    /// Borrow as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        // Read the owner so the foreign allocation is an actual use, not only a
        // field the drop glue happens to free.
        let _keep_alive = Arc::strong_count(&self.owner);
        if self.len == 0 {
            return &[];
        }
        // SAFETY: constructor contract — ptr valid for len while owner alive.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Length.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl F64Buffer {
    /// Owned buffer.
    #[must_use]
    pub fn owned(values: impl Into<Arc<[f64]>>) -> Self {
        Self { inner: F64BufferInner::Owned(values.into()), nan_free: false }
    }

    /// Foreign-backed buffer.
    #[must_use]
    pub fn foreign(buf: ForeignF64Buffer) -> Self {
        Self { inner: F64BufferInner::Foreign(buf), nan_free: false }
    }

    /// Whether the values are known to hold no `NaN`.
    ///
    /// Set once a full scan found none (the buffer is immutable, so the fact is
    /// carried by clones and by columns re-wrapping this buffer). `false` means
    /// unknown, not "holds a `NaN`".
    #[must_use]
    pub const fn is_nan_free(&self) -> bool {
        self.nan_free
    }

    /// Record that a full scan found no `NaN`.
    pub(crate) const fn mark_nan_free(&mut self) {
        self.nan_free = true;
    }

    /// Whether this buffer borrows foreign memory.
    #[must_use]
    pub const fn is_foreign(&self) -> bool {
        matches!(self.inner, F64BufferInner::Foreign(_))
    }

    /// Byte length of the logical values.
    #[must_use]
    pub fn nbytes(&self) -> u64 {
        (self.len() * core::mem::size_of::<f64>()) as u64
    }

    /// Borrow as slice.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        match &self.inner {
            F64BufferInner::Owned(v) => v.as_ref(),
            F64BufferInner::Foreign(f) => f.as_slice(),
        }
    }

    /// Length.
    #[must_use]
    pub fn len(&self) -> usize {
        match &self.inner {
            F64BufferInner::Owned(v) => v.len(),
            F64BufferInner::Foreign(f) => f.len(),
        }
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Deref for F64Buffer {
    type Target = [f64];

    fn deref(&self) -> &[f64] {
        self.as_slice()
    }
}

impl From<Arc<[f64]>> for F64Buffer {
    fn from(value: Arc<[f64]>) -> Self {
        Self::owned(value)
    }
}

impl From<Vec<f64>> for F64Buffer {
    fn from(value: Vec<f64>) -> Self {
        Self::owned(Arc::<[f64]>::from(value))
    }
}

impl PartialEq for F64Buffer {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

/// Keeps the allocation behind an offset foreign window alive.
#[cfg(test)]
#[derive(Debug)]
struct OffsetOwner(Vec<f64>);

#[cfg(test)]
impl ForeignBufferOwner for OffsetOwner {}

/// Contiguous offset window into `values`, kept as a foreign buffer.
///
/// A `NumPy` noncontiguous array is not imported. [`ForeignF64Buffer`] is the
/// ingestion path that already accepts a pointer and a length: an offset into a
/// larger allocation, which is the unit-stride window of that array. A strided
/// view does not grow a second snapshot type.
#[cfg(test)]
#[must_use]
pub(crate) fn foreign_offset_f64(values: Vec<f64>, start: usize, len: usize) -> F64Buffer {
    assert!(
        start.checked_add(len).is_some_and(|end| end <= values.len()),
        "offset window is outside the allocation"
    );
    let owner = Arc::new(OffsetOwner(values));
    let ptr = owner.0.as_ptr().wrapping_add(start);
    // SAFETY: `owner` keeps the `Vec<f64>` allocation alive and immutable. `start + len`
    // is in range, so `ptr` addresses `len` aligned `f64`s (or is empty and unread).
    // Moving the `Arc` does not reallocate.
    let foreign = unsafe { ForeignF64Buffer::new(owner, ptr, len) };
    F64Buffer::foreign(foreign)
}
