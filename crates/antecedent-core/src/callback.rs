//! Batch callbacks for intervention programs.
//!
//! A [`BatchCallback`] is a caller-supplied function from `k` numeric columns to one
//! numeric column, called once per batch (never per row). A program holds it through
//! [`CallbackCall`], which binds the callback to the argument expressions it reads, so
//! the callback's inputs are ordinary reads for dependency validation: circularity,
//! availability and history reads are decided by the expressions, not by the callback.
//!
//! A callback is a **runtime binding**: it executes wherever a program executes in this
//! process (simulation, counterfactual world evaluation), but it is not portable. Only
//! a callback declared with a recoverable implementation - a content digest of its
//! self-contained source, an entry point, and pinned dependency versions - can leave the
//! process, and then only as a [`RecoveryRecord`] the consumer re-verifies against the
//! implementation it supplies ([`RecoveryRecord::verify`]). The record never carries
//! code.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::canonical_encode::{CanonicalEncode, count};
use crate::identity::{CanonicalWriter, digest_from_lower_hex};
use crate::reason_code;
use crate::value_expr::ValueExpr;

/// The values a callback declares it can return.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum CallbackOutput {
    /// Any finite real number.
    Real,
    /// Finite reals in `[lower, upper]`.
    Range {
        /// Smallest admitted value.
        lower: f64,
        /// Largest admitted value.
        upper: f64,
    },
    /// Exactly these levels (sorted, unique, finite); a categorical or binary result.
    Levels(Arc<[f64]>),
}

impl CanonicalEncode for CallbackOutput {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::Real => {
                writer.word(1);
            }
            Self::Range { lower, upper } => {
                writer.word(2).real(*lower).real(*upper);
            }
            Self::Levels(levels) => {
                writer.word(3);
                writer.reals(levels);
            }
        }
    }
}

impl CallbackOutput {
    /// Levels, sorted and de-duplicated.
    #[must_use]
    pub fn levels(levels: impl IntoIterator<Item = f64>) -> Self {
        let mut levels: Vec<f64> = levels.into_iter().collect();
        levels.sort_by(f64::total_cmp);
        levels.dedup();
        Self::Levels(levels.into())
    }

    /// Whether the declaration itself is well formed.
    #[must_use]
    pub fn is_well_formed(&self) -> bool {
        match self {
            Self::Real => true,
            Self::Range { lower, upper } => {
                lower.is_finite() && upper.is_finite() && lower <= upper
            }
            Self::Levels(levels) => !levels.is_empty() && levels.iter().all(|l| l.is_finite()),
        }
    }

    /// Check a callback's output column against the declaration.
    ///
    /// # Errors
    ///
    /// [`CallbackError::NonFinite`] or [`CallbackError::OutsideSupport`] at the first
    /// offending row.
    pub fn validate(&self, output: &[f64]) -> Result<(), CallbackError> {
        if let Some(row) = output.iter().position(|v| !v.is_finite()) {
            return Err(CallbackError::NonFinite { row, value: output[row] });
        }
        let outside = match self {
            Self::Real => None,
            Self::Range { lower, upper } => output.iter().position(|v| v < lower || v > upper),
            Self::Levels(levels) => output.iter().position(|v| !levels.contains(v)),
        };
        match outside {
            Some(row) => Err(CallbackError::OutsideSupport {
                row,
                value: output[row],
                declared: self.to_string(),
            }),
            None => Ok(()),
        }
    }
}

impl fmt::Display for CallbackOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Real => f.write_str("real"),
            Self::Range { lower, upper } => write!(f, "range [{lower}, {upper}]"),
            Self::Levels(levels) => write!(f, "levels {levels:?}"),
        }
    }
}

/// A dependency pinned to an exact version.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct PinnedDependency {
    /// Distribution name.
    pub name: String,
    /// Exact version the callback was written and checked against.
    pub version: String,
}

/// Whether and how a callback can be recovered outside the process that built it.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CallbackImplementation {
    /// An in-process function. Not portable.
    Runtime,
    /// Self-contained source identified by content digest.
    Recoverable {
        /// Lower-case hex SHA-256 of the exact source text the callback runs.
        source_sha256: String,
        /// Name of the function in the source that is the callback.
        entry_point: String,
    },
}

impl CanonicalEncode for CallbackImplementation {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::Runtime => {
                writer.word(1);
            }
            Self::Recoverable { source_sha256, entry_point } => {
                writer.word(2).text(source_sha256).text(entry_point);
            }
        }
    }
}

/// The declaration of a callback: inputs, output and runtime dependencies.
#[derive(Clone, Debug, PartialEq)]
pub struct CallbackSpec {
    /// Callback name, used in refusals and records.
    pub name: String,
    /// One label per input, in call order; the arity is the label count.
    pub inputs: Arc<[String]>,
    /// Declared output.
    pub output: CallbackOutput,
    /// Dependencies pinned for the callback to behave as declared.
    pub dependencies: Arc<[PinnedDependency]>,
    /// Implementation identity.
    pub implementation: CallbackImplementation,
}

impl CallbackSpec {
    /// Whether the declaration is usable: a name, distinct input labels, a well-formed
    /// output and, when recoverable, a 64-digit digest and an entry point.
    #[must_use]
    pub fn is_well_formed(&self) -> bool {
        let distinct =
            self.inputs.iter().enumerate().all(|(i, label)| !self.inputs[..i].contains(label));
        let implementation = match &self.implementation {
            CallbackImplementation::Runtime => true,
            CallbackImplementation::Recoverable { source_sha256, entry_point } => {
                // Canonical identity text: lowercase hex only (uppercase is a different spelling).
                digest_from_lower_hex(source_sha256).is_some() && !entry_point.is_empty()
            }
        };
        !self.name.is_empty() && distinct && self.output.is_well_formed() && implementation
    }

    /// The record a consumer verifies, when the implementation is recoverable.
    #[must_use]
    pub fn recovery_record(&self) -> Option<RecoveryRecord> {
        match &self.implementation {
            CallbackImplementation::Runtime => None,
            CallbackImplementation::Recoverable { source_sha256, entry_point } => {
                Some(RecoveryRecord {
                    name: self.name.clone(),
                    inputs: Arc::clone(&self.inputs),
                    output: self.output.clone(),
                    source_sha256: source_sha256.clone(),
                    entry_point: entry_point.clone(),
                    dependencies: Arc::clone(&self.dependencies),
                })
            }
        }
    }
}

/// What an artifact may carry for a recoverable callback: identity, never code.
#[derive(Clone, Debug, PartialEq)]
pub struct RecoveryRecord {
    /// Callback name.
    pub name: String,
    /// Input labels in call order.
    pub inputs: Arc<[String]>,
    /// Declared output.
    pub output: CallbackOutput,
    /// Digest of the source the callback ran.
    pub source_sha256: String,
    /// Entry point within the source.
    pub entry_point: String,
    /// Pinned dependency versions.
    pub dependencies: Arc<[PinnedDependency]>,
}

/// A recovery record that does not match the implementation supplied for replay.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[error("callback {callback} does not match its recovery record: {field} differs")]
pub struct RecoveryMismatch {
    /// Callback name in the record.
    pub callback: String,
    /// First differing field.
    pub field: &'static str,
}

impl RecoveryMismatch {
    /// The mismatch as a reason-coded refusal.
    #[must_use]
    pub fn refusal(&self) -> CallbackRefusal {
        CallbackRefusal {
            code: reason_code!("policy_callback_unrecoverable"),
            message: format!(
                "policy_callback.recovery_mismatch: {self}; supply the implementation the record \
                 was made from, with the pinned dependency versions installed"
            ),
        }
    }
}

/// A reason-coded refusal about a callback: registered code and namespaced message.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallbackRefusal {
    /// Registered reason code.
    pub code: &'static str,
    /// Namespaced message, without the code prefix.
    pub message: String,
}

impl CallbackRefusal {
    /// The refusal as one message carrying its `reason=<code>:` prefix.
    #[must_use]
    pub fn coded_message(&self) -> String {
        format!("{}{}: {}", reason_code::PREFIX, self.code, self.message)
    }
}

impl RecoveryRecord {
    /// Verify that `supplied` is the implementation this record identifies.
    ///
    /// # Errors
    ///
    /// [`RecoveryMismatch`] naming the first differing field, or `implementation` when
    /// the supplied callback is not recoverable at all.
    pub fn verify(&self, supplied: &CallbackSpec) -> Result<(), RecoveryMismatch> {
        let mismatch = |field| RecoveryMismatch { callback: self.name.clone(), field };
        let Some(found) = supplied.recovery_record() else {
            return Err(mismatch("implementation"));
        };
        if found.name != self.name {
            return Err(mismatch("name"));
        }
        if found.source_sha256 != self.source_sha256 {
            return Err(mismatch("source_sha256"));
        }
        if found.entry_point != self.entry_point {
            return Err(mismatch("entry_point"));
        }
        if found.inputs != self.inputs {
            return Err(mismatch("inputs"));
        }
        if found.output != self.output {
            return Err(mismatch("output"));
        }
        if found.dependencies != self.dependencies {
            return Err(mismatch("dependencies"));
        }
        Ok(())
    }
}

/// Why a callback call failed.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum CallbackError {
    /// The callback itself raised; `message` is its own error text.
    #[error("{message}")]
    Raised {
        /// The callback's error message.
        message: String,
    },
    /// The callback returned the wrong number of values.
    #[error("returned {got} values; expected exactly {expected}")]
    WrongLength {
        /// Values returned.
        got: usize,
        /// Rows in the batch.
        expected: usize,
    },
    /// The callback returned something other than a one-dimensional real-valued column.
    #[error("returned {found}, but the declared output is a one-dimensional float64 column")]
    NotRealValued {
        /// What was returned (a dtype or type name).
        found: String,
    },
    /// A returned value is NaN or infinite.
    #[error("returned a non-finite value {value} at row {row}")]
    NonFinite {
        /// First offending row.
        row: usize,
        /// The value.
        value: f64,
    },
    /// A returned value is outside the declared output.
    #[error("returned {value} at row {row}, outside the declared {declared}")]
    OutsideSupport {
        /// First offending row.
        row: usize,
        /// The value.
        value: f64,
        /// The declaration.
        declared: String,
    },
}

impl CallbackError {
    /// Whether the failure is the callback raising, as opposed to a bad output.
    #[must_use]
    pub fn is_raised(&self) -> bool {
        matches!(self, Self::Raised { .. })
    }
}

/// A batch function of numeric columns, called once per batch.
pub trait BatchCallback: Send + Sync + fmt::Debug {
    /// The declaration: inputs, output, dependencies and implementation identity.
    fn spec(&self) -> &CallbackSpec;

    /// Fill `out` (one value per row) from `inputs` (one column per declared input, each
    /// `out.len()` long). The caller validates `out` against the declared output.
    ///
    /// # Errors
    ///
    /// [`CallbackError::Raised`] when the callback fails, or a shape / dtype error when
    /// it returns something other than one real value per row.
    fn call(&self, inputs: &[&[f64]], out: &mut [f64]) -> Result<(), CallbackError>;
}

/// A callback applied to argument expressions. Equal only to itself (same callback
/// object, same arguments): two closures with the same declaration may differ in code.
#[derive(Clone, Debug)]
pub struct CallbackCall {
    callback: Arc<dyn BatchCallback>,
    args: Arc<[ValueExpr]>,
}

// A callback is invoked only through `&self` and its declaration is immutable, so a panic
// inside a call cannot leave the binding (callback handle plus argument expressions) in a
// state a later observer could see broken. Declaring this keeps every type that can hold a
// program (`Intervention`, `CausalQuery`, ...) as panic-boundary safe as before callbacks
// existed; implementors that mutate shared state own the consistency of that state.
impl std::panic::UnwindSafe for CallbackCall {}
impl std::panic::RefUnwindSafe for CallbackCall {}

impl CanonicalEncode for CallbackCall {
    fn encode(&self, writer: &mut CanonicalWriter) {
        let spec = self.spec();
        writer.text(&spec.name);
        writer.texts(spec.inputs.as_ref());
        spec.output.encode(writer);
        writer.word(count(spec.dependencies.len()));
        for dependency in spec.dependencies.as_ref() {
            writer.text(&dependency.name).text(&dependency.version);
        }
        spec.implementation.encode(writer);
        writer.word(count(self.args().len()));
        for arg in self.args() {
            arg.encode(writer);
        }
    }
}

impl CallbackCall {
    /// Bind `callback` to `args`, one per declared input.
    #[must_use]
    pub fn new(callback: Arc<dyn BatchCallback>, args: impl Into<Arc<[ValueExpr]>>) -> Self {
        Self { callback, args: args.into() }
    }

    /// The callback.
    #[must_use]
    pub fn callback(&self) -> &Arc<dyn BatchCallback> {
        &self.callback
    }

    /// The declaration.
    #[must_use]
    pub fn spec(&self) -> &CallbackSpec {
        self.callback.spec()
    }

    /// Argument expressions in call order.
    #[must_use]
    pub fn args(&self) -> &[ValueExpr] {
        &self.args
    }

    /// The same call over different arguments.
    #[must_use]
    pub fn with_args(&self, args: impl Into<Arc<[ValueExpr]>>) -> Self {
        Self { callback: Arc::clone(&self.callback), args: args.into() }
    }
}

impl PartialEq for CallbackCall {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::addr_eq(Arc::as_ptr(&self.callback), Arc::as_ptr(&other.callback))
            && self.args == other.args
    }
}

impl Eq for CallbackCall {}

impl Hash for CallbackCall {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.callback.spec().name.hash(state);
        self.args.hash(state);
    }
}

/// Whether an expression can leave the process.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Portability {
    /// Closed-form: encodable as is.
    Portable,
    /// Contains callbacks, every one recoverable: encodable only as these records, which
    /// the consumer verifies against the implementations it supplies.
    Recoverable(Vec<RecoveryRecord>),
    /// Contains at least one runtime-bound callback: no durable encoding exists.
    RuntimeBound {
        /// Names of the runtime-bound callbacks.
        callbacks: Vec<String>,
    },
}

impl Portability {
    /// Combine two parts of one program: a runtime binding dominates.
    #[must_use]
    pub fn and(self, other: Self) -> Self {
        match (self, other) {
            (Self::RuntimeBound { mut callbacks }, Self::RuntimeBound { callbacks: more }) => {
                callbacks.extend(more);
                Self::RuntimeBound { callbacks }
            }
            (bound @ Self::RuntimeBound { .. }, _) | (_, bound @ Self::RuntimeBound { .. }) => {
                bound
            }
            (Self::Recoverable(mut records), Self::Recoverable(more)) => {
                records.extend(more);
                Self::Recoverable(records)
            }
            (records @ Self::Recoverable(_), _) | (_, records @ Self::Recoverable(_)) => records,
            _ => Self::Portable,
        }
    }

    /// The refusal an encoder returns for a runtime-bound program; `None` when the
    /// program is portable or recoverable.
    #[must_use]
    pub fn export_refusal(&self) -> Option<CallbackRefusal> {
        let Self::RuntimeBound { callbacks } = self else {
            return None;
        };
        Some(CallbackRefusal {
            code: reason_code!("policy_callback_not_portable"),
            message: format!(
                "policy_callback.runtime_binding: callback {} is an in-process function with \
                 no verifiable recoverable implementation; declare it from self-contained \
                 source with pinned dependencies, or keep the program inside this process",
                callbacks.join(", ")
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(implementation: CallbackImplementation) -> CallbackSpec {
        CallbackSpec {
            name: "f".into(),
            inputs: Arc::from(["x".to_string()]),
            output: CallbackOutput::Real,
            dependencies: Arc::from([PinnedDependency {
                name: "numpy".into(),
                version: "2.0".into(),
            }]),
            implementation,
        }
    }

    fn recoverable(digest: &str) -> CallbackImplementation {
        CallbackImplementation::Recoverable {
            source_sha256: digest.into(),
            entry_point: "f".into(),
        }
    }

    #[test]
    fn output_declarations_validate_finiteness_and_support() {
        assert!(CallbackOutput::Real.validate(&[1.0, -2.0]).is_ok());
        assert!(matches!(
            CallbackOutput::Real.validate(&[1.0, f64::NAN]),
            Err(CallbackError::NonFinite { row: 1, .. })
        ));
        assert!(matches!(
            CallbackOutput::Real.validate(&[f64::INFINITY]),
            Err(CallbackError::NonFinite { row: 0, .. })
        ));
        let range = CallbackOutput::Range { lower: 0.0, upper: 1.0 };
        assert!(range.validate(&[0.0, 0.5, 1.0]).is_ok());
        assert!(matches!(
            range.validate(&[0.5, 1.5]),
            Err(CallbackError::OutsideSupport { row: 1, .. })
        ));
        let levels = CallbackOutput::levels([2.0, 0.0, 2.0]);
        assert!(levels.validate(&[0.0, 2.0]).is_ok());
        assert!(matches!(
            levels.validate(&[0.0, 1.0]),
            Err(CallbackError::OutsideSupport { row: 1, .. })
        ));
    }

    #[test]
    fn malformed_declarations_are_not_well_formed() {
        assert!(!CallbackOutput::Range { lower: 1.0, upper: 0.0 }.is_well_formed());
        assert!(!CallbackOutput::levels([]).is_well_formed());
        assert!(!CallbackOutput::levels([f64::NAN]).is_well_formed());
        assert!(spec(CallbackImplementation::Runtime).is_well_formed());
        assert!(!spec(recoverable("abc")).is_well_formed());
        assert!(spec(recoverable(&"a".repeat(64))).is_well_formed());
        let mut twice = spec(CallbackImplementation::Runtime);
        twice.inputs = Arc::from(["x".to_string(), "x".to_string()]);
        assert!(!twice.is_well_formed());
    }

    #[test]
    fn a_recovery_record_verifies_only_the_implementation_it_names() {
        let digest = "a".repeat(64);
        let record = spec(recoverable(&digest)).recovery_record().unwrap();
        record.verify(&spec(recoverable(&digest))).unwrap();
        assert_eq!(
            record.verify(&spec(recoverable(&"b".repeat(64)))).unwrap_err().field,
            "source_sha256"
        );
        assert_eq!(
            record.verify(&spec(CallbackImplementation::Runtime)).unwrap_err().field,
            "implementation"
        );
        let mut other_pin = spec(recoverable(&digest));
        other_pin.dependencies =
            Arc::from([PinnedDependency { name: "numpy".into(), version: "2.1".into() }]);
        assert_eq!(record.verify(&other_pin).unwrap_err().field, "dependencies");
        let mut other_output = spec(recoverable(&digest));
        other_output.output = CallbackOutput::Range { lower: 0.0, upper: 1.0 };
        assert_eq!(record.verify(&other_output).unwrap_err().field, "output");
    }

    #[test]
    fn a_runtime_binding_dominates_portability() {
        let bound = Portability::RuntimeBound { callbacks: vec!["f".into()] };
        let record = spec(recoverable(&"a".repeat(64))).recovery_record().unwrap();
        let recoverable = Portability::Recoverable(vec![record]);
        assert_eq!(Portability::Portable.and(Portability::Portable), Portability::Portable);
        assert!(matches!(
            Portability::Portable.and(recoverable.clone()),
            Portability::Recoverable(r) if r.len() == 1
        ));
        let mixed = recoverable.and(bound);
        assert!(matches!(mixed, Portability::RuntimeBound { .. }));
        let refusal = mixed.export_refusal().unwrap();
        assert_eq!(refusal.code, "policy_callback_not_portable");
        assert!(refusal.coded_message().starts_with("reason=policy_callback_not_portable"));
        assert!(Portability::Portable.export_refusal().is_none());
    }
}
