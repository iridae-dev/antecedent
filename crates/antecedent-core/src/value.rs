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

    /// Interpret as an `f64` level when possible.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float64(v) => Some(*v),
            Self::Int64(v) => Some(*v as f64),
            Self::Bool(v) => Some(f64::from(u8::from(*v))),
            Self::Category(v) => Some(f64::from(*v)),
            Self::Label(_) => None,
        }
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

/// Renders each variant as: `Float64`/`Int64` via their native [`fmt::Display`](core::fmt::Display)
/// (e.g. `1.5`, `42`); `Bool` as `true`/`false`; `Category` as `cat(<code>)` (e.g. `cat(3)`);
/// and `Label` as the raw label text, unquoted.
impl core::fmt::Display for Value {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Float64(v) => write!(f, "{v}"),
            Self::Int64(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
            Self::Category(v) => write!(f, "cat({v})"),
            Self::Label(v) => write!(f, "{v}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_renders_each_variant() {
        assert_eq!(Value::Float64(1.5).to_string(), "1.5");
        assert_eq!(Value::Int64(42).to_string(), "42");
        assert_eq!(Value::Bool(true).to_string(), "true");
        assert_eq!(Value::Category(3).to_string(), "cat(3)");
        assert_eq!(Value::Label(Arc::from("foo")).to_string(), "foo");
    }
}
