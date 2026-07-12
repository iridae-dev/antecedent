//! Scalar and structured values used in queries and interventions.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

/// A concrete value assigned by an intervention or query contrast.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// Floating-point scalar.
    Float64(f64),
    /// Integer scalar.
    Int64(i64),
    /// Boolean.
    Bool(bool),
    /// Category code (raw u32; domain lives in the schema).
    Category(u32),
    /// Opaque label for diagnostics only (not used in hot paths as a key).
    Label(Arc<str>),
}

impl Value {
    /// Convenience for a float64 value.
    #[must_use]
    pub const fn f64(v: f64) -> Self {
        Self::Float64(v)
    }
}

impl Eq for Value {}

impl core::hash::Hash for Value {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            Self::Float64(v) => v.to_bits().hash(state),
            Self::Int64(v) => v.hash(state),
            Self::Bool(v) => v.hash(state),
            Self::Category(v) => v.hash(state),
            Self::Label(v) => v.hash(state),
        }
    }
}
