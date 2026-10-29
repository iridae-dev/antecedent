//! Data-layer errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;
use std::borrow::Cow;
use std::sync::Arc;

use antecedent_core::diagnostic::RouteNativeDiagnostic;
use antecedent_core::{SamplingDesignError, SchemaError, VariableId};

use crate::unbalanced_panel::PanelPreparationError;

/// What is wrong with a validity bitmap or the row map applied to it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum ValidityFault {
    /// A row map names a row past the bitmap.
    RowMapExceedsBitmap,
    /// The backing buffer holds fewer bytes than the bitmap length needs.
    BufferTooShort,
    /// The mask view refused the buffer.
    MaskViewRejectedBuffer,
}

impl fmt::Display for ValidityFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::RowMapExceedsBitmap => "row map exceeds bitmap",
            Self::BufferTooShort => "validity buffer too short",
            Self::MaskViewRejectedBuffer => "mask view rejected buffer",
        })
    }
}

/// Errors from data construction, lookup, or materialization.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DataError {
    /// Schema/data length mismatch.
    LengthMismatch {
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
        /// Context.
        context: &'static str,
    },
    /// Unknown variable in this table.
    UnknownVariable {
        /// Requested id.
        id: VariableId,
    },
    /// Column type does not match the requested view.
    TypeMismatch {
        /// Variable id.
        id: VariableId,
        /// Expected type label.
        expected: &'static str,
    },
    /// A validity bitmap or its row map is malformed.
    InvalidValidity {
        /// What is wrong with it.
        fault: ValidityFault,
    },
    /// Row selection produced an empty sample.
    EmptySelection {
        /// Explanation.
        context: &'static str,
    },
    /// Temporal gather requires a complete series (no missing values or masked rows).
    IncompleteSeries {
        /// Offending variable, when the gap is column-specific.
        id: Option<VariableId>,
        /// Explanation.
        message: &'static str,
    },
    /// Ad-hoc argument refusal raised by a caller outside the data layer. The data layer itself
    /// raises the typed variants below, so a caller matches on a kind, not on this text.
    InvalidArgument {
        /// Explanation.
        message: String,
    },
    /// An input that must hold at least one element is empty.
    EmptyInput {
        /// The input that is empty.
        what: &'static str,
    },
    /// An index, position or range end lies at or past the bound of what exists.
    OutOfRange {
        /// What was addressed.
        what: &'static str,
        /// The offending index (the end of a range).
        index: usize,
        /// The exclusive upper bound it had to stay under.
        bound: usize,
    },
    /// A size, count or arithmetic result does not fit the integer type that must hold it.
    Overflow {
        /// The quantity that does not fit.
        what: &'static str,
    },
    /// Two inputs that must agree (schemas, identities, shapes, grids) differ.
    Mismatch {
        /// What disagrees.
        what: &'static str,
    },
    /// A value is outside what the operation accepts.
    InvalidValue {
        /// The offending value.
        what: &'static str,
        /// What the operation accepts.
        requirement: &'static str,
    },
    /// One variable's column or declaration breaks a requirement of the operation.
    InvalidColumn {
        /// Offending variable.
        id: VariableId,
        /// What the operation requires of it.
        requirement: &'static str,
    },
    /// A value exceeds the limit that applies to it.
    ExceedsLimit {
        /// The quantity over its limit.
        what: &'static str,
        /// Its value.
        value: u64,
        /// The limit.
        limit: u64,
    },
    /// A key that must be unique appears more than once.
    Duplicate {
        /// The repeated key.
        what: &'static str,
    },
    /// Values that must be in order are not.
    NotOrdered {
        /// What is out of order.
        what: &'static str,
    },
    /// The operation is not defined for this input, policy or column kind.
    Unsupported {
        /// The refused operation.
        operation: &'static str,
        /// Why it is not defined here.
        reason: Cow<'static, str>,
    },
    /// A precondition of the operation does not hold.
    Precondition {
        /// The refused operation.
        operation: &'static str,
        /// What it requires.
        requirement: &'static str,
    },
    /// A required named input is absent.
    Absent {
        /// The kind of input.
        what: &'static str,
        /// Its name.
        name: String,
    },
    /// A byte or memory budget refuses the operation.
    BudgetRefused {
        /// What needed the bytes.
        what: &'static str,
        /// Bytes the operation needs.
        bytes: u64,
        /// The budget in force, when the refusal names one.
        budget: Option<u64>,
    },
    /// A foreign library failed.
    External {
        /// The library.
        library: &'static str,
        /// Its message.
        detail: String,
    },
    /// An internal invariant did not hold.
    Internal {
        /// The invariant.
        what: &'static str,
    },
    /// A missing value was met on a route that has no declared missing-data policy.
    MissingPolicyRequired {
        /// The route that refused.
        route: &'static str,
        /// Column holding the first missing cell.
        variable: Arc<str>,
        /// Zero-based row of that cell.
        row: usize,
        /// Policies the route can execute.
        available: Vec<&'static str>,
    },
    /// A panel layout, preparation, design or lag was refused.
    Panel(PanelPreparationError),
    /// Observation weights are not a valid non-negative measure.
    InvalidWeights {
        /// Offending row, or `None` when the defect is the total (no positive mass).
        index: Option<usize>,
        /// Explanation.
        reason: &'static str,
    },
    /// A cooperative cancellation was observed before the operation finished.
    Cancelled,
    /// The declared sampling design refused the requested resampling or splitting rule.
    SamplingDesign(SamplingDesignError),
    /// Underlying schema error.
    Schema(SchemaError),
}

impl DataError {
    /// Route record when this error is a dependent series resampled by a rule that
    /// drops its dependence.
    #[must_use = "a failed dependent-resampling check is dropped if the record is discarded"]
    pub const fn route_diagnostic(&self) -> Option<RouteNativeDiagnostic> {
        match self {
            Self::SamplingDesign(error) => error.route_diagnostic(),
            _ => None,
        }
    }
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { expected, actual, context } => {
                write!(f, "{context}: expected length {expected}, got {actual}")
            }
            Self::UnknownVariable { id } => write!(f, "unknown variable {id}"),
            Self::TypeMismatch { id, expected } => {
                write!(f, "variable {id} is not of type {expected}")
            }
            Self::InvalidValidity { fault } => write!(f, "invalid validity: {fault}"),
            Self::EmptySelection { context } => write!(f, "empty selection: {context}"),
            Self::IncompleteSeries { id, message } => match id {
                Some(id) => write!(f, "incomplete series (variable {id}): {message}"),
                None => write!(f, "incomplete series: {message}"),
            },
            Self::InvalidArgument { message } => write!(f, "invalid argument: {message}"),
            Self::EmptyInput { what } => write!(f, "invalid argument: {what} must not be empty"),
            Self::OutOfRange { what, index, bound } => {
                write!(f, "invalid argument: {what} {index} is out of range (bound {bound})")
            }
            Self::Overflow { what } => {
                write!(f, "invalid argument: {what} does not fit the integer type that holds it")
            }
            Self::Mismatch { what } => write!(f, "invalid argument: {what} do not match"),
            Self::InvalidValue { what, requirement } => {
                write!(f, "invalid argument: {what} {requirement}")
            }
            Self::InvalidColumn { id, requirement } => {
                write!(f, "invalid argument: variable {id} {requirement}")
            }
            Self::ExceedsLimit { what, value, limit } => {
                write!(f, "invalid argument: {what} {value} exceeds the limit {limit}")
            }
            Self::Duplicate { what } => write!(f, "invalid argument: duplicate {what}"),
            Self::NotOrdered { what } => write!(f, "invalid argument: {what} are not in order"),
            Self::Unsupported { operation, reason } => {
                write!(f, "invalid argument: {operation} is not supported: {reason}")
            }
            Self::Precondition { operation, requirement } => {
                write!(f, "invalid argument: {operation} requires {requirement}")
            }
            Self::Absent { what, name } => write!(f, "invalid argument: no {what} named {name}"),
            Self::BudgetRefused { what, bytes, budget: Some(budget) } => write!(
                f,
                "invalid argument: {what} needs {bytes} bytes, which exceeds the budget {budget}"
            ),
            Self::BudgetRefused { what, bytes, budget: None } => write!(
                f,
                "invalid argument: {what} needs {bytes} bytes, which the execution's memory \
                 budget does not admit"
            ),
            Self::External { library, detail } => {
                write!(f, "invalid argument: {library}: {detail}")
            }
            Self::Internal { what } => write!(f, "invalid argument: internal invariant: {what}"),
            Self::MissingPolicyRequired { route, variable, row, available } => write!(
                f,
                "invalid argument: {route} found a missing value in column `{variable}` at row \
                 {row}, which the declared missing-data policy `error` refuses; declare one of: {}",
                available.join(", ")
            ),
            Self::Panel(error) => write!(f, "invalid argument: {error}"),
            Self::InvalidWeights { index: Some(i), reason } => {
                write!(f, "invalid weights: row {i} is {reason}")
            }
            Self::InvalidWeights { index: None, reason } => write!(f, "invalid weights: {reason}"),
            Self::Cancelled => write!(f, "cancelled"),
            Self::SamplingDesign(error) => write!(f, "invalid argument: {error}"),
            Self::Schema(error) => write!(f, "schema error: {error}"),
        }
    }
}

/// Crate-private constructors of the typed argument refusals, so a call site names the kind
/// and the subject in one line.
impl DataError {
    pub(crate) const fn empty(what: &'static str) -> Self {
        Self::EmptyInput { what }
    }

    pub(crate) const fn out_of_range(what: &'static str, index: usize, bound: usize) -> Self {
        Self::OutOfRange { what, index, bound }
    }

    pub(crate) const fn overflow(what: &'static str) -> Self {
        Self::Overflow { what }
    }

    pub(crate) const fn mismatch(what: &'static str) -> Self {
        Self::Mismatch { what }
    }

    pub(crate) const fn invalid_value(what: &'static str, requirement: &'static str) -> Self {
        Self::InvalidValue { what, requirement }
    }

    pub(crate) const fn invalid_column(id: VariableId, requirement: &'static str) -> Self {
        Self::InvalidColumn { id, requirement }
    }

    pub(crate) const fn exceeds_limit(what: &'static str, value: u64, limit: u64) -> Self {
        Self::ExceedsLimit { what, value, limit }
    }

    pub(crate) const fn duplicate(what: &'static str) -> Self {
        Self::Duplicate { what }
    }

    pub(crate) const fn not_ordered(what: &'static str) -> Self {
        Self::NotOrdered { what }
    }

    pub(crate) fn unsupported(
        operation: &'static str,
        reason: impl Into<Cow<'static, str>>,
    ) -> Self {
        Self::Unsupported { operation, reason: reason.into() }
    }

    pub(crate) const fn precondition(operation: &'static str, requirement: &'static str) -> Self {
        Self::Precondition { operation, requirement }
    }

    pub(crate) fn absent(what: &'static str, name: impl Into<String>) -> Self {
        Self::Absent { what, name: name.into() }
    }

    pub(crate) const fn budget_refused(
        what: &'static str,
        bytes: u64,
        budget: Option<u64>,
    ) -> Self {
        Self::BudgetRefused { what, bytes, budget }
    }

    pub(crate) fn external(library: &'static str, detail: impl ToString) -> Self {
        Self::External { library, detail: detail.to_string() }
    }

    pub(crate) const fn internal(what: &'static str) -> Self {
        Self::Internal { what }
    }
}

impl std::error::Error for DataError {}

impl From<SchemaError> for DataError {
    fn from(err: SchemaError) -> Self {
        Self::Schema(err)
    }
}
