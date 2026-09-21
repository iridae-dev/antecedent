//! Scalar and structured values used in queries and interventions.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

/// Private-use marker for a symbolic intervention coordinate (`do(V)` with
/// unspecified level). Stored as [`Value::Label`] so it participates in
/// `Eq`/`Hash` (unlike `f64::NAN`, which breaks hash-consing).
const SYMBOLIC_INTERVENTION_LABEL: &str = "\u{E000}antecedent.symbolic_intervention";

/// A concrete value assigned by an intervention or query contrast.
#[derive(Clone, Debug)]
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

    /// Symbolic intervention placeholder (`do(V)` with no concrete level).
    ///
    /// This is the in-memory dual of the wire form's `symbolic: bool`. It must
    /// not be confused with a genuine non-finite float level.
    #[must_use]
    pub fn symbolic_intervention() -> Self {
        Self::Label(Arc::from(SYMBOLIC_INTERVENTION_LABEL))
    }

    /// Whether this value is the symbolic-intervention marker.
    #[must_use]
    pub fn is_symbolic_intervention(&self) -> bool {
        matches!(self, Self::Label(s) if s.as_ref() == SYMBOLIC_INTERVENTION_LABEL)
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

    /// Reject non-finite (and non-concrete) levels for hard interventions.
    ///
    /// A non-finite float is a validation error, not a symbolic wildcard.
    ///
    /// # Errors
    ///
    /// [`InterventionValueError::NonFinite`] for NaN/±∞ floats;
    /// [`InterventionValueError::NotConcrete`] for labels (including the
    /// symbolic marker) that are not concrete assignment levels.
    pub fn validate_concrete_intervention_level(&self) -> Result<(), InterventionValueError> {
        match self {
            Self::Float64(v) if !v.is_finite() => Err(InterventionValueError::NonFinite),
            Self::Label(_) => Err(InterventionValueError::NotConcrete),
            Self::Float64(_) | Self::Int64(_) | Self::Bool(_) | Self::Category(_) => Ok(()),
        }
    }
}

/// Errors from validating a concrete intervention level.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum InterventionValueError {
    /// Float level is NaN or infinite.
    #[error("intervention level must be finite")]
    NonFinite,
    /// Value is not a concrete numeric/category/bool level.
    #[error("intervention level is not concrete")]
    NotConcrete,
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            // Bit equality so NaN payloads compare equal and Eq stays sound.
            (Self::Float64(a), Self::Float64(b)) => a.to_bits() == b.to_bits(),
            (Self::Int64(a), Self::Int64(b)) => a == b,
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Category(a), Self::Category(b)) => a == b,
            (Self::Label(a), Self::Label(b)) => a == b,
            _ => false,
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
/// symbolic intervention as `·`; and `Label` as the raw label text, unquoted.
impl core::fmt::Display for Value {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Float64(v) => write!(f, "{v}"),
            Self::Int64(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
            Self::Category(v) => write!(f, "cat({v})"),
            Self::Label(v) if v.as_ref() == SYMBOLIC_INTERVENTION_LABEL => write!(f, "·"),
            Self::Label(v) => write!(f, "{v}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn display_renders_each_variant() {
        assert_eq!(Value::Float64(1.5).to_string(), "1.5");
        assert_eq!(Value::Int64(42).to_string(), "42");
        assert_eq!(Value::Bool(true).to_string(), "true");
        assert_eq!(Value::Category(3).to_string(), "cat(3)");
        assert_eq!(Value::Label(Arc::from("foo")).to_string(), "foo");
        assert_eq!(Value::symbolic_intervention().to_string(), "·");
    }

    #[test]
    fn symbolic_intervention_marker_is_eq_stable() {
        let a = Value::symbolic_intervention();
        let b = Value::symbolic_intervention();
        assert!(a.is_symbolic_intervention());
        assert_eq!(a, b);
        let mut set = HashSet::new();
        assert!(set.insert(a));
        assert!(!set.insert(b));
    }

    #[test]
    fn nan_is_not_the_symbolic_marker() {
        let nan = Value::f64(f64::NAN);
        assert!(!nan.is_symbolic_intervention());
        assert_ne!(nan, Value::symbolic_intervention());
    }

    #[test]
    fn non_finite_intervention_level_is_validation_error() {
        assert_eq!(
            Value::f64(f64::NAN).validate_concrete_intervention_level(),
            Err(InterventionValueError::NonFinite)
        );
        assert_eq!(
            Value::f64(f64::INFINITY).validate_concrete_intervention_level(),
            Err(InterventionValueError::NonFinite)
        );
        assert_eq!(
            Value::symbolic_intervention().validate_concrete_intervention_level(),
            Err(InterventionValueError::NotConcrete)
        );
        assert!(Value::f64(1.0).validate_concrete_intervention_level().is_ok());
        assert!(Value::Int64(0).validate_concrete_intervention_level().is_ok());
    }

    #[test]
    fn signed_zero_eq_and_hash_agree() {
        use std::hash::{Hash, Hasher};
        let hash = |v: &Value| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            v.hash(&mut h);
            h.finish()
        };
        let (pos, neg) = (Value::f64(0.0), Value::f64(-0.0));
        // Equality is by bits, exactly like the hash and the identity encoding.
        assert_ne!(pos, neg);
        assert_ne!(hash(&pos), hash(&neg));
        assert_eq!(hash(&pos), hash(&Value::f64(0.0)));
    }

    #[test]
    fn float_eq_uses_bits_so_nan_matches_itself() {
        let a = Value::f64(f64::NAN);
        let b = Value::f64(f64::NAN);
        assert_eq!(a, b);
        let mut set = HashSet::new();
        assert!(set.insert(a));
        assert!(!set.insert(b));
    }
}
