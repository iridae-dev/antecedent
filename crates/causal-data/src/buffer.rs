//! Contiguous f64 column buffers — owned or foreign-backed (DESIGN.md §5.2).
//!
//! Foreign buffers never expose Arrow types; the owner is type-erased.
//! Unsafe code here is limited to pointer→slice projection under an owner
//! lifetime contract (DESIGN.md §29).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(unsafe_code)]

use std::ops::Deref;
use std::sync::Arc;

/// Contiguous `f64` values for a column.
#[derive(Clone, Debug)]
pub struct F64Buffer {
    inner: F64BufferInner,
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

impl Drop for ForeignF64Buffer {
    fn drop(&mut self) {
        // Touch owner so clippy/rustc know it is semantically used for lifetime.
        let _ = Arc::as_ptr(&self.owner);
    }
}

/// Type-erased owner that keeps foreign memory valid.
pub trait ForeignBufferOwner: Send + Sync + std::fmt::Debug {}

impl<T: Send + Sync + std::fmt::Debug + 'static> ForeignBufferOwner for T {}

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
        Self { inner: F64BufferInner::Owned(values.into()) }
    }

    /// Foreign-backed buffer.
    #[must_use]
    pub fn foreign(buf: ForeignF64Buffer) -> Self {
        Self { inner: F64BufferInner::Foreign(buf) }
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
